# -*- coding: utf-8 -*-
"""
任务四（整体情感）：文档 6.3 SentimentClassifier

接口：
    train(texts, labels) -> None
    predict(text) -> (label, confidence)
    evaluate(X_test, y_test) -> Dict[str, float]
    save(path) / load(path)

同时提供 TF-IDF + SVM 传统机器学习基线（任务四要求对比两条技术路线）。
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

LABELS: List[str] = ["negative", "neutral", "positive"]
label2id: Dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}
id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                           labels: Sequence[str] = LABELS) -> Dict:
    """统一评估指标：准确率 / 宏平均 F1 / 加权 F1 / 各类别 P-R-F1 / 混淆矩阵。"""
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0)
    per_class = {
        lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
              "f1": round(float(f1[i]), 4), "support": int(support[i])}
        for i, lab in enumerate(labels)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "labels": list(labels),
    }


class _TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[str], tokenizer, max_len: int):
        self.items = []
        for text, lab in zip(texts, labels):
            enc = tokenizer(text, max_length=max_len, truncation=True,
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


class SentimentClassifier:
    """基于中文预训练语言模型的整体情感分类器（BERT + 线性分类头）。"""

    def __init__(self, model_name: str = "hfl/chinese-bert-wwm-ext", num_labels: int = 3,
                 max_len: int = 64, device: Optional[torch.device] = None):
        from transformers import BertForSequenceClassification, BertTokenizerFast
        self.model_name = model_name
        self.max_len = max_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels).to(self.device)

    # ---------------- 训练 ----------------
    def train(self, texts: List[str], labels: List[str], epochs: int = 2,
              batch_size: int = 16, lr: float = 2e-5, balanced_loss: bool = True,
              class_weight_power: float = 1.0, verbose: bool = True) -> List[float]:
        """
        文档接口 train(texts, labels)。

        balanced_loss: 是否使用类别加权交叉熵。影评整体情感中“中性”样本偏少且语义模糊，
        加权可避免模型直接把中性样本判成正/负，从而提升宏平均 F1。
        class_weight_power: 权重 = (总数 / (类别数 × 该类样本数)) ** power，
        1.0 为完全逆频率，0.5 为逆频率开方（更温和）。
        """
        ds = _TextDataset(texts, labels, self.tokenizer, self.max_len)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        weight = None
        if balanced_loss:
            counts = {lab: max(list(labels).count(lab), 1) for lab in LABELS}
            total = sum(counts.values())
            weight = torch.tensor(
                [(total / (len(LABELS) * counts[lab])) ** class_weight_power for lab in LABELS],
                dtype=torch.float).to(self.device)
            if verbose:
                print("  类别权重: " + ", ".join(
                    f"{lab}={w:.2f}" for lab, w in zip(LABELS, weight.tolist())))
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

    # ---------------- 预测 ----------------
    def predict_proba(self, text: str) -> List[float]:
        self.model.eval()
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=self.max_len).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=1)[0].cpu().tolist()

    def predict(self, text: str) -> Tuple[str, float]:
        """文档接口 predict(text) -> (label, confidence)。"""
        probs = self.predict_proba(text)
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return id2label[idx], float(probs[idx])

    def predict_batch(self, texts: Sequence[str]) -> List[Tuple[str, float]]:
        return [self.predict(t) for t in texts]

    # ---------------- 评估 ----------------
    def evaluate(self, X_test: Sequence[str], y_test: Sequence[str], verbose: bool = True) -> Dict:
        """文档接口 evaluate(X_test, y_test) -> Dict[str, float]。"""
        y_pred = [self.predict(t)[0] for t in X_test]
        metrics = classification_metrics(list(y_test), y_pred)
        if verbose:
            print(f"  准确率={metrics['accuracy']:.4f}  宏平均F1={metrics['macro_f1']:.4f}  "
                  f"加权F1={metrics['weighted_f1']:.4f}")
            for lab, m in metrics["per_class"].items():
                print(f"    {lab:<9} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} support={m['support']}")
        return metrics

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, max_len: int = 64, device: Optional[torch.device] = None) -> "SentimentClassifier":
        from transformers import BertForSequenceClassification, BertTokenizerFast
        obj = cls.__new__(cls)
        obj.model_name = path
        obj.max_len = max_len
        obj.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.tokenizer = BertTokenizerFast.from_pretrained(path)
        obj.model = BertForSequenceClassification.from_pretrained(path).to(obj.device)
        return obj


class TFIDFSVMBaseline:
    """传统机器学习基线：jieba 分词 + TF-IDF + 线性 SVM（任务四对比实验）。"""

    def __init__(self, tokenizer_fn, ngram_range: Tuple[int, int] = (1, 2), max_features: int = 50000):
        self.tokenizer_fn = tokenizer_fn
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenizer_fn, token_pattern=None, ngram_range=ngram_range,
            max_features=max_features, sublinear_tf=True,
        )
        self.model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TFIDFSVMBaseline":
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        return self

    def predict(self, texts: Sequence[str]) -> List[str]:
        return list(self.model.predict(self.vectorizer.transform(texts)))

    def decision_scores(self, texts: Sequence[str]):
        return self.model.decision_function(self.vectorizer.transform(texts))

    def evaluate(self, texts: Sequence[str], labels: Sequence[str], verbose: bool = True) -> Dict:
        pred = self.predict(texts)
        metrics = classification_metrics(list(labels), pred)
        if verbose:
            print(f"  准确率={metrics['accuracy']:.4f}  宏平均F1={metrics['macro_f1']:.4f}  "
                  f"加权F1={metrics['weighted_f1']:.4f}")
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)
        except Exception:
            # 分词函数若是 lambda（不可序列化），退化为“参数 + 词表 + IDF”方式保存
            params = {k: v for k, v in self.vectorizer.get_params().items() if k != "tokenizer"}
            with open(path, "wb") as f:
                pickle.dump({"model": self.model, "vectorizer_params": params,
                             "vocabulary": self.vectorizer.vocabulary_,
                             "idf": self.vectorizer.idf_}, f)
            print("    [提示] 分词函数不可序列化，TF-IDF 已改用“参数+词表+IDF”方式保存")

    @classmethod
    def load(cls, path: str, tokenizer_fn) -> "TFIDFSVMBaseline":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        inst = cls.__new__(cls)
        inst.tokenizer_fn = tokenizer_fn
        inst.model = payload["model"]
        if "vectorizer" in payload:
            inst.vectorizer = payload["vectorizer"]
        else:  # 重建 TfidfVectorizer（词表与 IDF 来自训练结果）
            import scipy.sparse as sp
            from sklearn.feature_extraction.text import TfidfTransformer
            vec = TfidfVectorizer(tokenizer=tokenizer_fn, **payload["vectorizer_params"])
            vec.vocabulary_ = payload["vocabulary"]
            vec.fixed_vocabulary_ = True
            params = payload["vectorizer_params"]
            idf = payload["idf"]
            vec._tfidf = TfidfTransformer(
                norm=params.get("norm", "l2"), use_idf=params.get("use_idf", True),
                smooth_idf=params.get("smooth_idf", True),
                sublinear_tf=params.get("sublinear_tf", False))
            vec._tfidf.idf_ = idf
            vec._tfidf._idf_diag = sp.diags(idf, offsets=0, shape=(len(idf), len(idf)),
                                            format="csr", dtype=np.float64)
            inst.vectorizer = vec
        return inst
