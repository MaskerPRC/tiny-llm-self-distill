"""
通用 Transformer 蒸馏训练脚本
支持 TinyBERT / MiniLM / DistilBERT 等 Hugging Face 模型的微调与 ONNX 导出
"""

import argparse
import json
import os
import sys
import io
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train(args):
    print(f"[训练] 加载数据: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    raw_labels = [item['label'] for item in data]

    unique_labels = sorted(set(raw_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    labels = [label2id[l] for l in raw_labels]

    labels_path = os.path.join(args.output_dir, 'labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(unique_labels, f, ensure_ascii=False)

    label2id_path = os.path.join(args.output_dir, 'label2id.json')
    with open(label2id_path, 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False)

    print(f"[训练] 标签映射: {label2id}")
    print(f"[训练] 数据量: {len(texts)}, 标签数: {len(unique_labels)}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    print(f"[训练] 训练集: {len(train_texts)}, 验证集: {len(val_texts)}")

    print(f"[训练] 加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=int(args.num_labels),
        ignore_mismatched_sizes=True,
    )

    tokenizer.save_pretrained(args.output_dir)

    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[训练] 设备: {device}")
    model.to(device)

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model')
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch['labels'].cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_path)
            print(f"  [BEST] Best model saved (Val Acc: {val_acc:.4f})")

    print(f"\n[训练] 最佳验证准确率: {best_val_acc:.4f}")

    print("[导出] 导出 ONNX 模型...")
    model.eval()
    model.to('cpu')

    dummy_input = tokenizer(
        "测试输入文本", return_tensors='pt',
        max_length=args.max_length, padding='max_length', truncation=True
    )

    onnx_path = os.path.join(args.output_dir, 'model.onnx')

    input_names = ['input_ids', 'attention_mask']
    if 'token_type_ids' in dummy_input:
        input_names.append('token_type_ids')

    dynamic_axes = {name: {0: 'batch_size', 1: 'sequence'} for name in input_names}
    dynamic_axes['output'] = {0: 'batch_size'}

    torch.onnx.export(
        model,
        tuple(dummy_input[k] for k in input_names),
        onnx_path,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[导出] ONNX 模型: {onnx_path} ({onnx_size_mb:.1f} MB)")

    report = classification_report(val_true, val_preds, target_names=unique_labels, output_dict=True)

    metrics = {
        'accuracy': best_val_acc,
        'f1_weighted': val_f1,
        'model_size_mb': onnx_size_mb,
        'num_labels': len(unique_labels),
        'train_samples': len(train_texts),
        'val_samples': len(val_texts),
        'epochs': args.epochs,
        'per_class': {
            label: {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1': report[label]['f1-score'],
            }
            for label in unique_labels if label in report
        },
    }

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"METRICS_JSON:{json.dumps(metrics)}")
    print(f"MODEL_PATH:{best_model_path}")
    print(f"ONNX_PATH:{onnx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer 分类模型训练')
    parser.add_argument('--data_path', required=True, help='训练数据 JSON 路径')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--model_name', default='huawei-noah/TinyBERT_General_4L_312D')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()
    train(args)
