"""
单文本回归/打分训练脚本
数据格式: [{text, score}] score 为 0.0~1.0 的浮点数
导出 ONNX 回归模型
"""
import argparse, json, os, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


class RegressionDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                   max_length=max_length, return_tensors='pt')
        self.scores = torch.tensor(scores, dtype=torch.float)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.scores[idx]
        return item


def train(args):
    print(f"[Regression] Loading: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [d['text'] for d in data]
    scores = [float(d['score']) for d in data]

    score_range = {'min': min(scores), 'max': max(scores), 'mean': np.mean(scores)}
    print(f"[Regression] Data: {len(data)}, Score range: [{score_range['min']:.2f}, {score_range['max']:.2f}]")

    with open(os.path.join(args.output_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    with open(os.path.join(args.output_dir, 'task_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'is_regression': True, 'score_range': score_range}, f, indent=2)

    train_texts, val_texts, train_scores, val_scores = train_test_split(
        texts, scores, test_size=args.val_split, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1, ignore_mismatched_sizes=True)
    model.config.problem_type = 'regression'

    tokenizer.save_pretrained(args.output_dir)
    if hasattr(tokenizer, 'get_vocab'):
        with open(os.path.join(args.output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer.get_vocab(), f, ensure_ascii=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = RegressionDataset(train_texts, train_scores, tokenizer, args.max_length)
    val_ds = RegressionDataset(val_texts, val_scores, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_corr = -1.0
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
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits.squeeze(-1)
                preds.extend(logits.cpu().numpy())
                trues.extend(batch['labels'].cpu().numpy())

        from scipy.stats import pearsonr
        corr, _ = pearsonr(trues, preds)
        mse = np.mean((np.array(trues) - np.array(preds)) ** 2)
        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(train_loader):.4f} | Pearson: {corr:.4f} | MSE: {mse:.4f}")

        if corr > best_corr:
            best_corr = corr
            model.save_pretrained(best_model_path)
            print(f"  [BEST] Saved (Pearson: {corr:.4f})")

    # ONNX export
    model.eval().to('cpu')
    dummy = tokenizer("test", return_tensors='pt', max_length=args.max_length,
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
        'accuracy': float(best_corr), 'pearson': float(best_corr), 'mse': float(mse),
        'model_size_mb': onnx_size, 'score_range': score_range,
        'train_samples': len(train_texts), 'val_samples': len(val_texts), 'epochs': args.epochs,
    }

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
    p.add_argument('--num_labels', type=int, default=1)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=2e-5)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--val_split', type=float, default=0.2)
    train(p.parse_args())
