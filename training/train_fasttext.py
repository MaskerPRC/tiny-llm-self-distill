"""
FastText 风格的轻量分类器训练脚本
使用简单的词袋 + 线性层实现极小模型（<2MB），导出为 JSON 权重供 Node.js 直接加载
"""

import argparse
import json
import os
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np


def tokenize(text):
    text = text.lower().strip()
    tokens = re.findall(r'[\w\u4e00-\u9fff]+', text)
    ngrams = []
    for t in tokens:
        ngrams.append(t)
    for i in range(len(tokens) - 1):
        ngrams.append(f"{tokens[i]}_{tokens[i+1]}")
    return ngrams


def train(args):
    print(f"[FastText] 加载数据: {args.data_path}")
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

    print(f"[FastText] 标签: {unique_labels}, 数据量: {len(texts)}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.val_split, random_state=42, stratify=labels
    )

    word_counts = Counter()
    for text in train_texts:
        word_counts.update(tokenize(text))

    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.most_common(args.vocab_size)) if count >= 2}
    print(f"[FastText] 词表大小: {len(vocab)}")

    num_labels = len(unique_labels)
    weights = defaultdict(lambda: np.zeros(num_labels))
    bias = np.zeros(num_labels)

    lr = args.learning_rate

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(train_texts))
        correct = 0
        total = 0
        total_loss = 0

        for idx in indices:
            text = train_texts[idx]
            label = train_labels[idx]
            tokens = tokenize(text)

            score = np.copy(bias)
            active_tokens = []
            for token in tokens:
                if token in vocab:
                    score += weights[token]
                    active_tokens.append(token)

            if len(active_tokens) > 0:
                score /= len(active_tokens)

            exp_scores = np.exp(score - np.max(score))
            probs = exp_scores / exp_scores.sum()

            total_loss += -np.log(probs[label] + 1e-10)
            pred = np.argmax(probs)
            correct += int(pred == label)
            total += 1

            grad = probs.copy()
            grad[label] -= 1.0

            if len(active_tokens) > 0:
                for token in active_tokens:
                    weights[token] -= lr * grad / len(active_tokens)
            bias -= lr * grad

        train_acc = correct / total
        avg_loss = total_loss / total

        val_preds = []
        for text in val_texts:
            tokens = tokenize(text)
            score = np.copy(bias)
            active = [t for t in tokens if t in vocab]
            for token in active:
                score += weights[token]
            if len(active) > 0:
                score /= len(active)
            val_preds.append(np.argmax(score))

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        lr *= 0.95

    weights_dict = {}
    for word, w_vec in weights.items():
        if np.any(np.abs(w_vec) > 0.01):
            label_weights = {}
            for i, label in enumerate(unique_labels):
                if abs(w_vec[i]) > 0.01:
                    label_weights[label] = float(w_vec[i])
            if label_weights:
                weights_dict[word] = label_weights

    weights_path = os.path.join(args.output_dir, 'weights.json')
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, ensure_ascii=False)

    model_info = {
        'type': 'fasttext',
        'vocab_size': len(vocab),
        'num_labels': num_labels,
        'bias': bias.tolist(),
    }
    model_path = os.path.join(args.output_dir, 'model_info.json')
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    model_size = os.path.getsize(weights_path) / (1024 * 1024)

    report = classification_report(val_labels, val_preds, target_names=unique_labels, output_dict=True)
    metrics = {
        'accuracy': val_acc,
        'f1_weighted': val_f1,
        'model_size_mb': model_size,
        'num_labels': num_labels,
        'vocab_size': len(vocab),
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

    print(f"\n[FastText] 模型大小: {model_size:.2f} MB")
    print(f"METRICS_JSON:{json.dumps(metrics)}")
    print(f"MODEL_PATH:{weights_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastText 分类器训练')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name', default='fasttext')
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--vocab_size', type=int, default=50000)
    args = parser.parse_args()
    train(args)
