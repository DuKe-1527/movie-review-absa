# -*- coding: utf-8 -*-
"""
任务三：方面术语抽取 Aspect Term Extraction（文档 6.2 AspectExtractor）

把“方面术语抽取”建模为 BIO 序列标注任务，提供两条技术路线：
    基础：BiLSTM + CRF（从零训练，CPU 秒级）
    进阶：BERT / RoBERTa + CRF（中文预训练模型微调）

接口（与任务书一致）：
    extract(text) -> List[AspectSentiment]
    train(dataset) -> None
    save(path) -> None
    load(path) -> None
    另附 evaluate(dataset) -> Dict[str, float]   # 实体级 精确率/召回率/F1
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchcrf
from torch.utils.data import DataLoader, Dataset

from .taxonomy import (
    ASPECT_CATEGORY_LIST,
    ASPECT_LEXICON,
    UNKNOWN_CATEGORY,
    aspect_category,
    lexicon_extract,
)

BIO_LABELS: List[str] = ["O", "B-ASPECT", "I-ASPECT"]
label2id_bio: Dict[str, int] = {lab: i for i, lab in enumerate(BIO_LABELS)}
id2label_bio: Dict[int, str] = {i: lab for lab, i in label2id_bio.items()}


def text_to_bio(text: str, aspect_terms: Sequence[str]) -> List[str]:
    """按字符生成 BIO 标签序列；重叠命中时保留先出现的方面术语。"""
    tags = ["O"] * len(text)
    for term in sorted(set(t for t in aspect_terms if t), key=len, reverse=True):
        start = text.find(term)
        while start != -1:
            if all(tag == "O" for tag in tags[start:start + len(term)]):
                tags[start] = "B-ASPECT"
                for i in range(start + 1, start + len(term)):
                    tags[i] = "I-ASPECT"
            start = text.find(term, start + 1)
    return tags


def char_tags_to_terms(text: str, char_tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """字符级 BIO 标签 → 方面术语（起止位置, 术语文本）。"""
    spans, cur = [], None
    for i, tag in enumerate(char_tags):
        if tag == "B-ASPECT":
            if cur:
                spans.append(cur)
            cur = [i, i + 1]
        elif tag == "I-ASPECT" and cur:
            cur[1] = i + 1
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [(s, e, text[s:e]) for s, e in spans if e > s]


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
class BertASPDataset(Dataset):
    """BERT 输入的 token 级 BIO 数据集（中文 BERT 基本 1 字符 = 1 token）。"""

    def __init__(self, samples: List[Dict], tokenizer, max_len: int = 64):
        self.items = []
        for sample in samples:
            text = sample["text"]
            terms = [a["term"] if isinstance(a, dict) else a for a in sample.get("aspects", [])]
            bio = text_to_bio(text, terms)
            enc = tokenizer(
                text, max_length=max_len, truncation=True, padding="max_length",
                return_offsets_mapping=True,
            )
            labels = []
            for (s, e) in enc["offset_mapping"]:
                if s == e:  # 特殊符号 [CLS]/[SEP]/padding
                    labels.append(label2id_bio["O"])
                else:
                    labels.append(label2id_bio[bio[s]] if s < len(bio) else label2id_bio["O"])
            self.items.append({
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(enc["token_type_ids"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


class CharASPDataset(Dataset):
    """BiLSTM 输入的字符级 BIO 数据集（字符 ID 化，定长 padding）。"""

    def __init__(self, samples: List[Dict], char2id: Dict[str, int], max_len: int = 64):
        self.items = []
        for sample in samples:
            text = sample["text"][:max_len]
            terms = [a["term"] if isinstance(a, dict) else a for a in sample.get("aspects", [])]
            bio = text_to_bio(text, terms)
            ids = [char2id.get(ch, char2id["<unk>"]) for ch in text]
            labels = [label2id_bio[t] for t in bio]
            pad = max_len - len(ids)
            self.items.append({
                "input_ids": torch.tensor(ids + [char2id["<pad>"]] * pad, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.long),
                "labels": torch.tensor(labels + [label2id_bio["O"]] * pad, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class BiLSTMCRF(nn.Module):
    """基础版序列标注模型：字符 Embedding + BiLSTM + CRF。"""

    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int = 128,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = torchcrf.CRF(num_tags, batch_first=True)

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x))

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        logits = self.logits(input_ids, attention_mask)
        mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
            return loss, logits
        return self.crf.decode(logits, mask=mask)


class BertCRFTagger(nn.Module):
    """进阶版序列标注模型：中文 BERT + CRF。"""

    def __init__(self, model_name: str, num_tags: int):
        super().__init__()
        from transformers import BertConfig, BertForTokenClassification
        try:
            self.bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_tags)
        except OSError:
            # 本地模型目录只保存了 config.json + state_dict（未导出权重文件）时的加载路径
            config = BertConfig.from_pretrained(model_name, num_labels=num_tags)
            self.bert = BertForTokenClassification(config)
        self.crf = torchcrf.CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs):
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = out.logits
        mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
            return loss, logits
        return self.crf.decode(logits, mask=mask)


# --------------------------------------------------------------------------
# 对外主类
# --------------------------------------------------------------------------
class AspectExtractor:
    """方面术语抽取器：model_type='bert' 使用 BERT-CRF，'bilstm' 使用 BiLSTM-CRF。"""

    def __init__(self, model_type: str = "bert", model_name: str = "hfl/chinese-bert-wwm-ext",
                 max_len: int = 64, device: Optional[torch.device] = None):
        from transformers import BertTokenizerFast
        self.model_type = model_type
        self.model_name = model_name
        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.char2id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        if model_type == "bert":
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            self.model = BertCRFTagger(model_name, len(BIO_LABELS)).to(self.device)
        else:
            self.tokenizer = None
            self.model = None

    # ---------------- 训练 ----------------
    def _ensure_bilstm(self, texts: Sequence[str]) -> None:
        for ch in set("".join(texts)):
            if ch not in self.char2id:
                self.char2id[ch] = len(self.char2id)
        if self.model is None:
            self.model = BiLSTMCRF(len(self.char2id), len(BIO_LABELS)).to(self.device)

    def train(self, dataset: List[Dict], epochs: int = 2, batch_size: int = 16,
              lr: Optional[float] = None, verbose: bool = True) -> List[float]:
        """文档接口 train(dataset)；dataset 形如 [{'text': str, 'aspects': [{'term': str}]}]"""
        if lr is None:  # 预训练模型用微调学习率，从零训练的 BiLSTM 需要更大幅度
            lr = 3e-5 if self.model_type == "bert" else 1e-3
        if self.model_type == "bert":
            ds = BertASPDataset(dataset, self.tokenizer, self.max_len)
        else:
            self._ensure_bilstm([d["text"] for d in dataset])
            ds = CharASPDataset(dataset, self.char2id, self.max_len)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        history = []
        for epoch in range(epochs):
            total = 0.0
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, _ = self.model(**batch)
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

    # ---------------- 预测 ----------------
    def _char_tags(self, text: str) -> List[str]:
        """预测字符级 BIO 标签序列。"""
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "bert":
                enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                     padding="max_length", max_length=self.max_len,
                                     return_offsets_mapping=True)
                offsets = enc.pop("offset_mapping")[0].tolist()
                enc = {k: v.to(self.device) for k, v in enc.items()}
                pred = self.model(**enc)[0]
                tags = ["O"] * len(text)
                for idx, tag_id in enumerate(pred):
                    s, e = offsets[idx] if idx < len(offsets) else (0, 0)
                    if s == e:
                        continue
                    for c in range(s, min(e, len(text))):
                        if tags[c] == "O":
                            tags[c] = id2label_bio[tag_id]
                return tags
            ids = [self.char2id.get(ch, self.char2id["<unk>"]) for ch in text[: self.max_len]]
            pad = self.max_len - len(ids)
            batch = {
                "input_ids": torch.tensor([ids + [0] * pad], dtype=torch.long).to(self.device),
                "attention_mask": torch.tensor([[1] * len(ids) + [0] * pad], dtype=torch.long).to(self.device),
            }
            pred = self.model(**batch)[0]
            tags = [id2label_bio[i] for i in pred[: len(ids)]]
            return tags + ["O"] * pad

    def extract(self, text: str, use_lexicon_fallback: bool = False) -> List[Dict]:
        """文档接口 extract(text) -> List[AspectSentiment]（含类别、情感占位、置信度）。"""
        text = text or ""
        results: List[Dict] = []
        if self.model is not None and text:
            tags = self._char_tags(text)
            seen = set()
            for _, _, term in char_tags_to_terms(text, tags):
                term = term.strip()
                if len(term) < 1 or term in seen:
                    continue
                seen.add(term)
                results.append({
                    "aspect_term": term,
                    "aspect_category": aspect_category(term),
                    "sentiment": None,
                    "confidence": 0.0,
                    "source": "model",
                })
        if not results and use_lexicon_fallback:
            results = lexicon_extract(text)
        return results

    # ---------------- 评估 ----------------
    def evaluate(self, dataset: List[Dict], verbose: bool = True) -> Dict[str, float]:
        """实体级（精确匹配）精确率 / 召回率 / F1。"""
        tp = fp = fn = 0
        for sample in dataset:
            text = sample["text"]
            gold = {a["term"] if isinstance(a, dict) else a for a in sample.get("aspects", [])}
            gold = {g for g in gold if g and g in text}
            pred = {r["aspect_term"] for r in self.extract(text)}
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
        if verbose:
            print(f"  实体级 P={precision:.4f}  R={recall:.4f}  F1={f1:.4f} "
                  f"(tp={tp}, fp={fp}, fn={fn})")
        return metrics

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        meta = {"model_type": self.model_type, "model_name": self.model_name,
                "max_len": self.max_len, "bio_labels": BIO_LABELS}
        if self.model_type == "bert":
            torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
            self.tokenizer.save_pretrained(path)
            # 保存 BertForTokenClassification 的 config.json，保证离线可加载
            self.model.bert.config.save_pretrained(path)
            with open(os.path.join(path, "aspect_config.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            torch.save({"state_dict": self.model.state_dict(), "char2id": self.char2id, **meta},
                       os.path.join(path, "bilstm_crf.pt"))

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "AspectExtractor":
        if os.path.exists(os.path.join(path, "aspect_config.json")):
            with open(os.path.join(path, "aspect_config.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            obj = cls(model_type="bert", model_name=path, max_len=meta.get("max_len", 64), device=device)
            obj.model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=obj.device))
            obj.model.to(obj.device)
            return obj
        ckpt = torch.load(os.path.join(path, "bilstm_crf.pt"), map_location="cpu")
        obj = cls.__new__(cls)
        obj.model_type = "bilstm"
        obj.model_name = ckpt.get("model_name", "bilstm-crf")
        obj.max_len = ckpt.get("max_len", 64)
        obj.device = device or torch.device("cpu")
        obj.tokenizer = None
        obj.char2id = ckpt["char2id"]
        obj.model = BiLSTMCRF(len(obj.char2id), len(BIO_LABELS)).to(obj.device)
        obj.model.load_state_dict(ckpt["state_dict"])
        return obj
