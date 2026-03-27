"""
句子对相似度/关系分类训练脚本
数据格式: [{text_a, text_b, label}] (分类) 或 [{text_a, text_b, score}] (回归)
导出 ONNX 模型
"""
import argparse, json, os, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length, is_regression=False):
        texts_a = [p['text_a'] for p in pairs]
        texts_b = [p['text_b'] for p in pairs]
        self.encodings = tokenizer(texts_a, texts_b, truncation=True,
                                   padding='max_length', max_length=max_length, return_tensors='pt')
        if is_regression:
            self.labels = torch.tensor([p['score'] for p in pairs], dtype=torch.float)
        else:
            self.labels = torch.tensor([p['label_id'] for p in pairs], dtype=torch.long)
        self.is_regression = is_regression

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train(args):
    print(f"[Similarity] Loading: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    is_regression = 'score' in data[0] and 'label' not in data[0]

    if is_regression:
        num_labels = 1
        unique_labels = []
        print(f"[Similarity] Regression mode, data: {len(data)}")
    else:
        unique_labels = sorted(set(d['label'] for d in data))
        label2id = {l: i for i, l in enumerate(unique_labels)}
        for d in data:
            d['label_id'] = label2id[d['label']]
        num_labels = len(unique_labels)
        print(f"[Similarity] Classification mode, labels: {unique_labels}, data: {len(data)}")

    with open(os.path.join(args.output_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(unique_labels, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'task_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'is_regression': is_regression, 'num_labels': num_labels}, f)

    train_data, val_data = train_test_split(data, test_size=args.val_split, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    if is_regression:
        model.config.problem_type = 'regression'

    tokenizer.save_pretrained(args.output_dir)
    if hasattr(tokenizer, 'get_vocab'):
        with open(os.path.join(args.output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer.get_vocab(), f, ensure_ascii=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = PairDataset(train_data, tokenizer, args.max_length, is_regression)
    val_ds = PairDataset(val_data, tokenizer, args.max_length, is_regression)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_metric = -1.0
    best_model_path = os.path.join(args.output_dir, 'best_model')
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                if is_regression:
                    preds_all.extend(logits.squeeze(-1).cpu().numpy())
                    labels_all.extend(batch['labels'].cpu().numpy())
                else:
                    preds_all.extend(logits.argmax(dim=-1).cpu().numpy())
                    labels_all.extend(batch['labels'].cpu().numpy())

        if is_regression:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(labels_all, preds_all)
            mse = np.mean((np.array(labels_all) - np.array(preds_all)) ** 2)
            metric = corr
            print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(train_loader):.4f} | Pearson: {corr:.4f} | MSE: {mse:.4f}")
        else:
            acc = accuracy_score(labels_all, preds_all)
            f1 = f1_score(labels_all, preds_all, average='weighted')
            metric = acc
            print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        if metric > best_metric:
            best_metric = metric
            model.save_pretrained(best_model_path)
            print(f"  [BEST] Saved (metric: {metric:.4f})")

    # ONNX export
    model.eval().to('cpu')
    dummy = tokenizer("text a", "text b", return_tensors='pt', max_length=args.max_length,
                       padding='max_length', truncation=True)
    input_names = ['input_ids', 'attention_mask']
    if 'token_type_ids' in dummy:
        input_names.append('token_type_ids')

    onnx_path = os.path.join(args.output_dir, 'model.onnx')
    dynamic_axes = {n: {0: 'batch', 1: 'seq'} for n in input_names}
    dynamic_axes['output'] = {0: 'batch'}

    torch.onnx.export(model, tuple(dummy[k] for k in input_names), onnx_path,
                       input_names=input_names, output_names=['output'],
                       dynamic_axes=dynamic_axes, opset_version=14, do_constant_folding=True)

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    metrics = {
        'accuracy': float(best_metric), 'model_size_mb': onnx_size,
        'is_regression': is_regression, 'num_labels': num_labels,
        'train_samples': len(train_data), 'val_samples': len(val_data), 'epochs': args.epochs,
    }
    if not is_regression:
        metrics['f1_weighted'] = float(f1)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"METRICS_JSON:{json.dumps(metrics)}")
    print(f"MODEL_PATH:{best_model_path}")
    print(f"ONNX_PATH:{onnx_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model_name', default='hfl/rbt3')
    p.add_argument('--num_labels', type=int, default=2)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--val_split', type=float, default=0.2)
    train(p.parse_args())
