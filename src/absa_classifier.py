# -*- coding: utf-8 -*-
"""
任务四（方面级情感）：ABSA 方面级情感分类器

输入形式为 [CLS] 评论文本 [SEP] 方面术语 [SEP] 的句对，输出 negative / neutral / positive。
接口与 SentimentClassifier 保持一致：
    train(texts, aspects, labels) / predict(text, aspect) / evaluate(...) / save / load
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .sentiment_classifier import LABELS, classification_metrics, id2label, label2id


class _ABSADataset(Dataset):
    def __init__(self, texts: Sequence[str], aspects: Sequence[str], labels: Sequence[str],
                 tokenizer, max_len: int):
        self.items = []
        for text, aspect, lab in zip(texts, aspects, labels):
            enc = tokenizer(text, aspect, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            self.items.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "token_type_ids": enc["token_type_ids"][0],
                "labels": torch.tensor(label2id[lab], dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class ABSASentimentClassifier:
    """方面级情感分类器（文本-方面句对输入）。"""

    def __init__(self, model_name: str = "hfl/chinese-bert-wwm-ext", num_labels: int = 3,
                 max_len: int = 96, device: Optional[torch.device] = None):
        from transformers import BertForSequenceClassification, BertTokenizerFast
        self.model_name = model_name
        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels).to(self.device)

    def train(self, texts: List[str], aspects: List[str], labels: List[str], epochs: int = 2,
              batch_size: int = 16, lr: float = 2e-5, balanced_loss: bool = True,
              class_weight_power: float = 1.0, verbose: bool = True) -> List[float]:
        """
        balanced_loss: 方面情感样本中中性类极少（本例约 5%），
        采用逆频率加权（class_weight_power=1.0），避免模型完全忽略中性类导致宏平均 F1 偏低。
        """
        ds = _ABSADataset(texts, aspects, labels, self.tokenizer, self.max_len)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        weight = None
        if balanced_loss:
            counts = {lab: max(labels.count(lab), 1) for lab in LABELS}
            total = sum(counts.values())
            weight = torch.tensor(
                [(total / (len(LABELS) * counts[lab])) ** class_weight_power for lab in LABELS],
                dtype=torch.float).to(self.device)
            if verbose:
                print("  类别权重（逆频率）: " +
                      ", ".join(f"{lab}={w:.2f}" for lab, w in zip(LABELS, weight.tolist())))
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        self.model.train()
        history = []
        for epoch in range(epochs):
            total = 0.0
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels_t = batch.pop("labels")
                logits = self.model(**batch).logits
                loss = loss_fn(logits, labels_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += float(loss.item())
            avg = total / max(len(dl), 1)
            history.append(avg)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
        return history

    def predict_proba(self, text: str, aspect: str) -> List[float]:
        self.model.eval()
        enc = self.tokenizer(text, aspect, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=self.max_len).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=1)[0].cpu().tolist()

    def predict(self, text: str, aspect: str) -> Tuple[str, float]:
        probs = self.predict_proba(text, aspect)
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return id2label[idx], float(probs[idx])

    def evaluate(self, texts: Sequence[str], aspects: Sequence[str], labels: Sequence[str],
                 verbose: bool = True) -> Dict:
        y_pred = [self.predict(t, a)[0] for t, a in zip(texts, aspects)]
        metrics = classification_metrics(list(labels), y_pred)
        if verbose:
            print(f"  准确率={metrics['accuracy']:.4f}  宏平均F1={metrics['macro_f1']:.4f}  "
                  f"加权F1={metrics['weighted_f1']:.4f}")
            for lab, m in metrics["per_class"].items():
                print(f"    {lab:<9} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} support={m['support']}")
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, max_len: int = 96,
             device: Optional[torch.device] = None) -> "ABSASentimentClassifier":
        from transformers import BertForSequenceClassification, BertTokenizerFast
        obj = cls.__new__(cls)
        obj.model_name = path
        obj.max_len = max_len
        obj.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.tokenizer = BertTokenizerFast.from_pretrained(path)
        obj.model = BertForSequenceClassification.from_pretrained(path).to(obj.device)
        return obj

    # 便于使用 LABELS 常量（negative / neutral / positive）
    labels: List[str] = LABELS
