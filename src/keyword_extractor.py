# -*- coding: utf-8 -*-
"""
任务五（关键词抽取）：文档 6.4 KeywordExtractor

接口：
    extract(text, top_k=5) -> List[str]                 单条评论关键词
    global_keywords(texts, top_k=20) -> List[str]       语料级高频关键词（TF-IDF）

内部实现三条技术路线并做加权融合：
    TF-IDF（jieba.analyse）/ TextRank（textrank4zh）/ RAKE（自实现）
"""
from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from textrank4zh import TextRank4Keyword

from .compat import patch_networkx_for_textrank
from .preprocessor import SENTIMENT_LEXICON

# textrank4zh 依赖 networkx<3 的接口，这里先做兼容处理
patch_networkx_for_textrank()

_SENT_SPLIT = re.compile(r"[，。！？；、,.!?;:\s]+")

# 低信息量词（程度副词、指代词等），关键词抽取时过滤
LOW_INFO_WORDS = {
    "这部电影", "这电影", "这个", "那个", "这么", "那么", "就是", "还是", "只是",
    "略显", "极具", "非常", "真的", "感觉", "有点", "比较", "特别", "确实", "简直",
    "而且", "但是", "不过", "其实", "所以", "因为", "如果", "虽然", "已经", "可能",
    "应该", "一直", "居然", "果然", "一部", "一下子", "一个",
}


class KeywordExtractor:
    """影评关键词 / 关键短语抽取器。"""

    def __init__(self, stopwords_path: str, user_dict_path: Optional[str] = None):
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as f:
                self.stopwords = {line.strip() for line in f if line.strip()}
        if user_dict_path and os.path.exists(user_dict_path):
            jieba.load_userdict(user_dict_path)
        self.tr4w = TextRank4Keyword()

    # ---------------- 单条评论 ----------------
    def _clean_terms(self, pairs: Sequence[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        out = []
        for word, weight in pairs:
            w = word.strip()
            if (len(w) < 2 and not w.isdigit()) or w in self.stopwords or w in LOW_INFO_WORDS:
                continue
            if re.fullmatch(r"[0-9\W_]+", w):
                continue
            out.append((w, float(weight)))
        return out[:top_k]

    def tfidf_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        return self._clean_terms(jieba.analyse.extract_tags(text, topK=top_k * 2, withWeight=True), top_k)

    def textrank_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        try:
            self.tr4w.analyze(text=text, lower=True, window=2)
            pairs = [(d["word"], d["weight"]) for d in self.tr4w.get_keywords(num=top_k * 2, word_min_len=2)]
        except Exception:
            pairs = []
        return self._clean_terms(pairs, top_k)

    def rake_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """RAKE：以停用词/标点切分候选短语，按词频与共现度打分。"""
        phrases = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
        candidates: List[List[str]] = []
        for phrase in phrases:
            words = [w for w in jieba.lcut(phrase) if w.strip() and w not in self.stopwords]
            if words:
                candidates.append(words)
        freq: Dict[str, int] = {}
        degree: Dict[str, int] = {}
        for words in candidates:
            for w in words:
                freq[w] = freq.get(w, 0) + 1
                degree[w] = degree.get(w, 0) + max(len(words) - 1, 1)
        scores: Dict[str, float] = {}
        for words in candidates:
            for w in set(words):
                if len(w) < 2 or re.fullmatch(r"[0-9\W_]+", w):
                    continue
                scores[w] = scores.get(w, 0.0) + (degree[w] / freq[w]) + freq[w]
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def extract_with_weights(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """三路融合：TF-IDF 0.4 + TextRank 0.4 + RAKE 0.2（各自归一化后加权）。"""

        def norm(pairs: List[Tuple[str, float]]) -> Dict[str, float]:
            if not pairs:
                return {}
            mx = max(w for _, w in pairs) or 1.0
            return {k: w / mx for k, w in pairs}

        tfidf = norm(self.tfidf_keywords(text, top_k * 3))
        textrank = norm(self.textrank_keywords(text, top_k * 3))
        rake = norm(self.rake_keywords(text, top_k * 3))
        merged: Dict[str, float] = {}
        # 短评场景下 TF-IDF 最稳定，TextRank 次之，RAKE 主要用于补充长短语
        for src, weight in ((tfidf, 0.5), (textrank, 0.3), (rake, 0.2)):
            for word, score in src.items():
                if word in LOW_INFO_WORDS:
                    continue
                merged[word] = merged.get(word, 0.0) + weight * score
        ordered = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:top_k]

    def extract(self, text: str, top_k: int = 5) -> List[str]:
        """文档接口 extract(text, top_k) -> List[str]。"""
        return [w for w, _ in self.extract_with_weights(text, top_k)]

    def extract_phrases(self, text: str, top_k: int = 3) -> List[str]:
        """关键短语抽取：优先输出电影领域中较长的专名/名词短语。"""
        phrases = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
        scored = []
        for p in phrases:
            words = [w for w in jieba.lcut(p) if w.strip()]
            if len(p) < 4 or not words:
                continue
            hits = sum(1 for w in words if w in SENTIMENT_LEXICON)
            score = len(p) * 0.2 + hits * 1.5
            scored.append((p, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    # ---------------- 语料级 ----------------
    def global_keywords(self, texts: Sequence[str], top_k: int = 20) -> List[str]:
        """文档接口 global_keywords(texts, top_k)：语料级 TF-IDF 关键词。"""
        vectorizer = TfidfVectorizer(
            tokenizer=lambda t: [w for w in jieba.lcut(t) if w.strip() and w not in self.stopwords
                                 and len(w) > 1],
            token_pattern=None, max_features=20000, sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(list(texts))
        terms = vectorizer.get_feature_names_out()
        weights = matrix.sum(axis=0).A1
        order = weights.argsort()[::-1][:top_k]
        return [str(terms[i]) for i in order]

    def global_keywords_with_weights(self, texts: Sequence[str], top_k: int = 20) -> List[Tuple[str, float]]:
        vectorizer = TfidfVectorizer(
            tokenizer=lambda t: [w for w in jieba.lcut(t) if w.strip() and w not in self.stopwords
                                 and len(w) > 1],
            token_pattern=None, max_features=20000, sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(list(texts))
        terms = vectorizer.get_feature_names_out()
        weights = matrix.sum(axis=0).A1
        order = weights.argsort()[::-1][:top_k]
        return [(str(terms[i]), float(weights[i])) for i in order]
