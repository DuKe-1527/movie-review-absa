# -*- coding: utf-8 -*-
"""
任务六：模型可解释性分析（文档 6.6 Explainer）

    1) LIMEExplainer        —— 词级 LIME，输出对预测贡献最大的词语
    2) OcclusionExplainer   —— 逐词遮挡（SHAP 思路的轻量替代，无需额外依赖）
    3) analyze_errors       —— 错误样本分类归因（反讽 / 否定 / 隐含方面 / 中性混淆 …）
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jieba
import numpy as np

from .sentiment_classifier import LABELS

SENTIMENT_ZH = {"positive": "正面", "negative": "负面", "neutral": "中性"}


class LIMEExplainer:
    """文档接口：explain(text, classifier) -> Dict（特征重要性）。"""

    def __init__(self, class_names: Sequence[str] = LABELS, num_features: int = 8,
                 num_samples: int = 200, word_level: bool = True, random_state: int = 42):
        import lime.lime_text
        self.class_names = list(class_names)
        self.num_features = num_features
        self.num_samples = num_samples
        split = (lambda x: [w for w in jieba.lcut(x) if w.strip()]) if word_level else r"\W+"
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=self.class_names, split_expression=split, random_state=random_state)

    def _wrapper(self, texts: Sequence[str], classifier) -> List[List[float]]:
        # LIME 内部会把返回值当二维数组切片，必须返回 numpy 数组（概率行和为 1）
        out = [list(classifier.predict_proba(t)) for t in texts]
        return np.asarray(out, dtype=float)

    def explain(self, text: str, classifier, num_features: Optional[int] = None) -> Dict:
        """对分类器在 text 上的预测做 LIME 解释。"""
        probs = classifier.predict_proba(text)
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        expl = self.explainer.explain_instance(
            text, lambda x: self._wrapper(x, classifier),
            labels=(pred_idx,),
            num_features=num_features or self.num_features,
            num_samples=self.num_samples,
        )
        weights = [(w, float(s)) for w, s in expl.as_list(label=pred_idx)]
        weights.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return {
            "text": text,
            "label": self.class_names[pred_idx],
            "probability": float(probs[pred_idx]),
            "distribution": {name: float(p) for name, p in zip(self.class_names, probs)},
            "feature_weights": weights,
            "positive_features": [w for w, s in weights if s > 0],
            "negative_features": [w for w, s in weights if s < 0],
            "method": f"LIME(word-level, num_samples={self.num_samples})",
        }


class OcclusionExplainer:
    """逐词遮挡解释：删除某词前后目标类别概率变化 = 该词的重要度（SHAP 替代方案）。"""

    def __init__(self, max_words: int = 15):
        self.max_words = max_words

    def explain(self, text: str, classifier) -> Dict:
        words = [w for w in jieba.lcut(text) if w.strip()]
        unique = list(dict.fromkeys(words))[: self.max_words]
        base = classifier.predict_proba(text)
        pred_idx = int(max(range(len(base)), key=lambda i: base[i]))
        effects: List[Tuple[str, float]] = []
        for word in unique:
            masked = text.replace(word, "")
            if not masked.strip():
                continue
            probs = classifier.predict_proba(masked)
            effects.append((word, float(base[pred_idx] - probs[pred_idx])))
        effects.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return {
            "text": text,
            "label": LABELS[pred_idx],
            "probability": float(base[pred_idx]),
            "distribution": {name: float(p) for name, p in zip(LABELS, base)},
            "feature_weights": effects,
            "positive_features": [w for w, s in effects if s > 0],
            "negative_features": [w for w, s in effects if s < 0],
            "method": "Occlusion(deletion)",
        }


def error_type(text: str, true_label: str, pred_label: str,
               irony: bool = False, negation: bool = False,
               aspect_terms: Optional[Sequence[str]] = None) -> str:
    """错误样本归因：给出错误类型标签（用于任务六错误案例分析）。"""
    aspects = list(aspect_terms or [])
    if irony:
        return "反讽/反话"
    if negation and len(text) > 8:
        return "否定/双重否定"
    if not aspects:
        return "隐含方面/无显式方面词"
    if len(text) > 60:
        return "长文本多观点冲突"
    if "neutral" in (true_label, pred_label):
        return "中性/弱情感混淆"
    return "极性判断相反"


def analyze_errors(records: Sequence[Dict]) -> Dict:
    """
    错误案例分析。

    records 每条形如：
        {"text": str, "true": str, "pred": str, "irony": bool,
         "negation": bool, "aspect_terms": List[str], "confidence": float}
    """
    cases, type_counter = [], {}
    for rec in records:
        if rec["true"] == rec["pred"]:
            continue
        etype = error_type(rec["text"], rec["true"], rec["pred"],
                           rec.get("irony", False), rec.get("negation", False),
                           rec.get("aspect_terms"))
        type_counter[etype] = type_counter.get(etype, 0) + 1
        cases.append({**rec, "error_type": etype})
    n = len(records)
    total_err = len(cases)
    return {
        "num_samples": n,
        "num_errors": total_err,
        "error_rate": round(total_err / n, 4) if n else 0.0,
        "type_counts": dict(sorted(type_counter.items(), key=lambda kv: kv[1], reverse=True)),
        "cases": cases,
    }


def plot_feature_importance(explanation: Dict, ax=None, title: Optional[str] = None):
    """可视化解释结果（正面贡献 / 负面贡献 分色）。"""
    import matplotlib.pyplot as plt
    weights = explanation["feature_weights"][::-1]
    words = [w for w, _ in weights]
    scores = [s for _, s in weights]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(2.4, 0.42 * len(words) + 1)))
    else:
        fig = ax.figure
    colors = ["#2E8B57" if s > 0 else "#C0392B" for s in scores]
    ax.barh(words, scores, color=colors)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel(f"对预测「{explanation['label']}」的贡献")
    ax.set_title(title or f"{explanation['method']}｜P({explanation['label']})="
                          f"{explanation['probability']:.3f}")
    for i, s in enumerate(scores):
        ax.text(s + (0.005 if s >= 0 else -0.005), i, f"{s:+.3f}",
                va="center", ha="left" if s >= 0 else "right", fontsize=8)
    fig.tight_layout()
    return fig
