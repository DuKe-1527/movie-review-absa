# -*- coding: utf-8 -*-
"""
任务五（观点摘要）：文档 6.5 OpinionSummarizer

接口：
    summarize(text, aspect_sentiments) -> str

实现：
    抽取式（默认）：方面覆盖度 + 情感强度 + TextRank 句子重要性 + 位置权重 综合打分
    生成式（可选）：调用大模型（ChatGLM/DeepSeek）生成凝练摘要，失败自动回退抽取式
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from textrank4zh import TextRank4Sentence

from .compat import patch_networkx_for_textrank
from .preprocessor import NEGATIVE_WORDS, POSITIVE_WORDS

patch_networkx_for_textrank()

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")

SENTIMENT_ZH = {"positive": "正面", "negative": "负面", "neutral": "中性"}


class OpinionSummarizer:
    """基于方面级情感结果的观点摘要生成器。"""

    def __init__(self, max_sentences: int = 2, llm_client=None):
        self.max_sentences = max_sentences
        self.llm_client = llm_client
        self.tr4s = TextRank4Sentence()

    # ---------------- 句子切分 ----------------
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        parts = [p.strip() for p in _SENT_SPLIT.split(str(text)) if p and p.strip()]
        return [p for p in parts if len(p) >= 2] or [str(text).strip()]

    def _textrank_scores(self, sentences: List[str]) -> List[float]:
        if len(sentences) < 2:
            return [1.0] * len(sentences)
        try:
            self.tr4s.analyze(text="。".join(sentences), lower=True)
            weight = {}
            for item in self.tr4s.get_key_sentences(num=len(sentences), sentence_min_len=2):
                weight[item["index"]] = item["weight"]
            scores = [weight.get(i, 0.0) for i in range(len(sentences))]
            mx = max(scores) or 1.0
            return [s / mx for s in scores]
        except Exception:
            return [1.0] * len(sentences)

    # ---------------- 句子打分 ----------------
    def sentence_scores(self, text: str, aspect_sentiments: Optional[Sequence[Dict]] = None) -> List[Tuple[str, float]]:
        sentences = self.split_sentences(text)
        aspects = list(aspect_sentiments or [])
        tr_scores = self._textrank_scores(sentences)
        scored: List[Tuple[str, float]] = []
        for idx, sent in enumerate(sentences):
            score = 0.6 * tr_scores[idx]
            covered = []
            for asp in aspects:
                term = asp.get("aspect_term") or asp.get("term") or ""
                if term and term in sent:
                    conf = float(asp.get("confidence") or 0.8)
                    score += 1.2 + conf
                    sentiment = asp.get("sentiment")
                    if sentiment == "positive" and any(w in sent for w in POSITIVE_WORDS):
                        score += 0.4
                    if sentiment == "negative" and any(w in sent for w in NEGATIVE_WORDS):
                        score += 0.4
                    covered.append(term)
            score += 0.15 * len(covered)
            score += 0.35 if idx == 0 else 0.0            # 首句往往概括主旨
            score -= 0.4 if len(sent) < 6 else 0.0        # 过短句子信息量低
            score -= 0.05 * max(len(sent) - 60, 0) / 10   # 过长句子轻微惩罚
            scored.append((sent, round(score, 4)))
        return scored

    # ---------------- 主接口 ----------------
    def summarize(self, text: str, aspect_sentiments: Optional[Sequence[Dict]] = None,
                  max_sentences: Optional[int] = None, use_llm: bool = False) -> str:
        """文档接口 summarize(text, aspect_sentiments) -> str。"""
        if use_llm and self.llm_client is not None:
            llm_summary = self.summarize_llm(text, aspect_sentiments)
            if llm_summary:
                return llm_summary
        k = max_sentences or self.max_sentences
        scored = self.sentence_scores(text, aspect_sentiments)
        top = sorted(range(len(scored)), key=lambda i: scored[i][1], reverse=True)[:k]
        chosen = [scored[i][0] for i in sorted(top)]
        # 结尾标点规范化：把分号/逗号结尾改为句号，避免出现“；。”这类拼接痕迹
        cleaned = []
        for sent in chosen:
            sent = sent.rstrip()
            if sent and sent[-1] in "；;，,、":
                sent = sent[:-1]
            cleaned.append(sent if sent.endswith(("。", "！", "？", "!", "?")) else sent + "。")
        summary = "".join(cleaned)
        return summary.strip()

    def summarize_llm(self, text: str, aspect_sentiments: Optional[Sequence[Dict]] = None) -> Optional[str]:
        """生成式摘要：让模型基于方面级情感结果写一句凝练观点摘要。"""
        if self.llm_client is None:
            return None
        aspect_desc = "；".join(
            f"{a.get('aspect_term') or a.get('term')}（{a.get('aspect_category', '')}）="
            f"{SENTIMENT_ZH.get(a.get('sentiment'), '未知')}"
            for a in (aspect_sentiments or [])
        ) or "无明显方面词"
        prompt = (
            "你是电影评论观点摘要助手。请根据评论原文与方面级情感分析结果，"
            "生成一段不超过 80 字的中文观点摘要，要求覆盖主要方面及其情感倾向，避免主观发挥。\n"
            f"评论原文：{text}\n方面级结果：{aspect_desc}\n只输出摘要正文。"
        )
        out = self.llm_client.chat([{"role": "user", "content": prompt}], max_tokens=200)
        return out.strip().replace("\n", "") if out else None

    def summarize_corpus(self, texts: Sequence[str], aspect_sentiments: Sequence[Dict],
                         top_k: int = 6) -> str:
        """多文档口碑摘要（扩展方向）：统计方面维度的口碑分布。"""
        stat: Dict[str, Dict[str, int]] = {}
        for asp in aspect_sentiments:
            cat = asp.get("aspect_category") or "其他"
            senti = asp.get("sentiment") or "neutral"
            stat.setdefault(cat, {"positive": 0, "negative": 0, "neutral": 0})
            stat[cat][senti] = stat[cat].get(senti, 0) + 1
        ranked = sorted(stat.items(),
                        key=lambda kv: sum(kv[1].values()), reverse=True)[:top_k]
        parts = []
        for cat, cnt in ranked:
            total = sum(cnt.values()) or 1
            best = max(cnt.items(), key=lambda kv: kv[1])[0]
            parts.append(f"{cat}以{SENTIMENT_ZH[best]}评价为主（{cnt[best]}/{total}）")
        total_docs = len(texts)
        return f"共分析 {total_docs} 条影评：" + "；".join(parts) + "。"
