# -*- coding: utf-8 -*-
"""
任务七（可视化）：文档 6.8 visualizer（情感分布、方面热力图、关键词图、错误类型分布…）

本环境未安装 wordcloud / seaborn，因此词云用“关键词条形图 + 气泡图”替代实现
（如后续安装 wordcloud，可在 plot_word_frequency_bubbles 旁按同样接口扩展真实词云）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["negative", "neutral", "positive"]
LABELS_ZH = {"negative": "负面", "neutral": "中性", "positive": "正面"}
COLORS = {"negative": "#C0392B", "neutral": "#7F8C8D", "positive": "#2E8B57"}
CATEGORY_ORDER = ["剧本剧情", "演员表演", "视觉效果", "导演水平",
                  "音乐音效", "叙事节奏", "台词文本", "整体体验", "其他"]


def set_chinese_font(font_paths: Optional[Sequence[str]] = None) -> str:
    """配置 matplotlib 中文字体（Windows: 微软雅黑 / 黑体；也支持项目内 ttf）。"""
    from matplotlib import font_manager
    candidates = list(font_paths or [])
    candidates += [
        os.environ.get("SIMHEI_FONT", ""),
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                matplotlib.rcParams["font.sans-serif"] = [name] + matplotlib.rcParams["font.sans-serif"]
                break
            except Exception:
                continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib.rcParams["font.sans-serif"][0]


def plot_sentiment_pie(counts: Dict[str, int], title: str = "整体情感分布", ax=None):
    """情感分布饼图。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
    else:
        fig = ax.figure
    labels = [LABELS_ZH[k] for k in LABELS if counts.get(k, 0) > 0]
    sizes = [counts.get(k, 0) for k in LABELS if counts.get(k, 0) > 0]
    colors = [COLORS[k] for k in LABELS if counts.get(k, 0) > 0]
    if not sizes:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        return fig
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white"})
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def plot_aspect_heatmap(aspect_sentiments: Sequence[Dict], title: str = "方面-情感热力图", ax=None):
    """方面类别 × 情感极性 热力图。"""
    matrix = np.zeros((len(CATEGORY_ORDER), len(LABELS)))
    for asp in aspect_sentiments:
        cat = asp.get("aspect_category") or "其他"
        senti = asp.get("sentiment") or "neutral"
        if cat not in CATEGORY_ORDER:
            cat = "其他"
        matrix[CATEGORY_ORDER.index(cat), LABELS.index(senti)] += 1
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 5.6))
    else:
        fig = ax.figure
    im = ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(LABELS)), [LABELS_ZH[x] for x in LABELS])
    ax.set_yticks(range(len(CATEGORY_ORDER)), CATEGORY_ORDER)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="方面数量")
    fig.tight_layout()
    return fig


def plot_aspect_frequency(aspect_sentiments: Sequence[Dict], top_k: int = 12,
                          title: str = "高频方面统计", ax=None):
    """高频方面统计柱状图（按情感分层堆叠）。"""
    from collections import defaultdict
    stat = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for asp in aspect_sentiments:
        stat[asp.get("aspect_category") or "其他"][asp.get("sentiment") or "neutral"] += 1
    ranked = sorted(stat.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:top_k]
    cats = [k for k, _ in ranked][::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(cats) + 2))
    else:
        fig = ax.figure
    left = np.zeros(len(cats))
    for lab in LABELS:
        vals = np.array([stat[c][lab] for c in cats], dtype=float)
        ax.barh(cats, vals, left=left, color=COLORS[lab], label=LABELS_ZH[lab])
        left += vals
    ax.set_title(title)
    ax.set_xlabel("方面数量")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_keyword_barh(pairs: Sequence[Tuple[str, float]], title: str = "关键词 / 短语",
                      ax=None, color: str = "#2C6E9B"):
    """关键词条形图（词云的可视化替代方案）。"""
    pairs = list(pairs)[::-1]
    words = [w for w, _ in pairs]
    scores = [float(s) for _, s in pairs]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 0.4 * len(words) + 1.6))
    else:
        fig = ax.figure
    ax.barh(words, scores, color=color)
    ax.set_title(title)
    for i, s in enumerate(scores):
        ax.text(s * 1.01, i, f"{s:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_word_frequency_bubbles(pairs: Sequence[Tuple[str, float]],
                                title: str = "语料关键词分布（气泡图）", ax=None):
    """词云替代方案：气泡大小映射词权重（无需 wordcloud 依赖）。"""
    rng = np.random.default_rng(42)
    n = len(pairs)
    xs = rng.uniform(0.05, 0.95, n)
    ys = rng.uniform(0.1, 0.9, n)
    weights = np.array([float(w) for _, w in pairs])
    sizes = 350 * weights / (weights.max() or 1.0) + 120
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.6))
    else:
        fig = ax.figure
    ax.scatter(xs, ys, s=sizes, c=weights, cmap="viridis", alpha=0.55, edgecolors="white")
    for (word, _), x, y in zip(pairs, xs, ys):
        ax.text(x, y, word, ha="center", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: Sequence[Sequence[int]], labels: Sequence[str] = LABELS,
                          title: str = "混淆矩阵", ax=None):
    """分类混淆矩阵热力图。"""
    cm = np.array(cm, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
    else:
        fig = ax.figure
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_error_types(type_counts: Dict[str, int], title: str = "错误类型分布", ax=None):
    """错误类型分布柱状图。"""
    items = sorted(type_counts.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(names) + 1.5))
    else:
        fig = ax.figure
    ax.barh(names, vals, color="#B5651D")
    ax.set_xlabel("样本数")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v * 1.01, i, str(v), va="center", fontsize=8)
    fig.tight_layout()
    return fig


def save_fig(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path
