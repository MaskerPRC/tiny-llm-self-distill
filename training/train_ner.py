"""
NER (Named Entity Recognition) 训练脚本
数据格式: [{text, entities: [{start, end, label}]}]
导出 ONNX token-classification 模型
"""
import argparse, json, os, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


def build_tag_list(entity_labels):
    tags = ['O']
    for label in sorted(entity_labels):
        tags.append(f'B-{label}')
        tags.append(f'I-{label}')
    return tags


def align_labels(text, entities, tokenizer, max_length, tag2id):
    encoding = tokenizer(text, truncation=True, padding='max_length',
                         max_length=max_length, return_offsets_mapping=True, return_tensors='pt')
    offsets = encoding.pop('offset_mapping')[0].tolist()
    label_ids = [tag2id['O']] * max_length

    for ent in entities:
        s, e, lbl = ent['start'], ent['end'], ent['label']
        first = True
        for idx, (os_, oe_) in enumerate(offsets):
            if os_ == oe_ == 0:
                continue
            if os_ >= s and oe_ <= e:
                tag = f'B-{lbl}' if first else f'I-{lbl}'
                label_ids[idx] = tag2id.get(tag, 0)
                first = False

    return {k: v.squeeze(0) for k, v in encoding.items()}, label_ids


class NERDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length, tag2id):
        self.items = []
        for ex in examples:
            enc, labels = align_labels(ex['text'], ex.get('entities', []), tokenizer, max_length, tag2id)
            enc['labels'] = torch.tensor(labels, dtype=torch.long)
            self.items.append(enc)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train(args):
    print(f"[NER] Loading: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entity_labels = set()
    for item in data:
        for ent in item.get('entities', []):
            entity_labels.add(ent['label'])

    tags = build_tag_list(entity_labels)
    tag2id = {t: i for i, t in enumerate(tags)}
    id2tag = {i: t for t, i in tag2id.items()}

    with open(os.path.join(args.output_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(tags, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'tag2id.json'), 'w', encoding='utf-8') as f:
        json.dump(tag2id, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, 'entity_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(sorted(entity_labels), f, ensure_ascii=False)

    print(f"[NER] Entity types: {sorted(entity_labels)}, Tags: {len(tags)}, Data: {len(data)}")

    train_data, val_data = train_test_split(data, test_size=args.val_split, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(tags), ignore_mismatched_sizes=True)

    tokenizer.save_pretrained(args.output_dir)
    if hasattr(tokenizer, 'get_vocab'):
        with open(os.path.join(args.output_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(tokenizer.get_vocab(), f, ensure_ascii=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"[NER] Device: {device}")

    train_ds = NERDataset(train_data, tokenizer, args.max_length, tag2id)
    val_ds = NERDataset(val_data, tokenizer, args.max_length, tag2id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    best_f1 = 0.0
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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch_dev).logits
                preds = logits.argmax(dim=-1).cpu().numpy()
                labs = batch['labels'].numpy()
                mask = labs != tag2id['O']
                for p_seq, l_seq, m_seq in zip(preds, labs, mask):
                    for p, l, m in zip(p_seq, l_seq, m_seq):
                        if m or l != tag2id['O']:
                            all_preds.append(p)
                            all_labels.append(l)

        if all_labels:
            from sklearn.metrics import f1_score as f1
            val_f1 = f1(all_labels, all_preds, average='weighted', zero_division=0)
            acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(len(all_labels), 1)
        else:
            val_f1, acc = 0.0, 0.0

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss/len(train_loader):.4f} | Entity Acc: {acc:.4f} | Entity F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(best_model_path)
            print(f"  [BEST] Saved (F1: {val_f1:.4f})")

    # ONNX export
    model.eval().to('cpu')
    dummy = tokenizer("test", return_tensors='pt', max_length=args.max_length, padding='max_length', truncation=True)
    input_names = ['input_ids', 'attention_mask']
    if 'token_type_ids' in dummy:
        input_names.append('token_type_ids')

    onnx_path = os.path.join(args.output_dir, 'model.onnx')
    dynamic_axes = {n: {0: 'batch', 1: 'seq'} for n in input_names}
    dynamic_axes['output'] = {0: 'batch', 1: 'seq'}

    torch.onnx.export(model, tuple(dummy[k] for k in input_names), onnx_path,
                       input_names=input_names, output_names=['output'],
                       dynamic_axes=dynamic_axes, opset_version=14, do_constant_folding=True)

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    metrics = {
        'accuracy': acc, 'f1_weighted': best_f1, 'model_size_mb': onnx_size,
        'num_tags': len(tags), 'entity_types': sorted(entity_labels),
        'train_samples': len(train_data), 'val_samples': len(val_data), 'epochs': args.epochs,
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
    p.add_argument('--num_labels', type=int, default=3)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=3e-5)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--val_split', type=float, default=0.2)
    train(p.parse_args())
