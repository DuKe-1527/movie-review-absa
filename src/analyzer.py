# -*- coding: utf-8 -*-
"""
文档 6.7 ReviewAnalyzerApp：把六个模块串成端到端流水线

    analyze(text) -> ReviewResult           单条评论完整分析
    analyze_batch(file_path) -> List[...]   批量文件分析（jsonl / csv）
    visualize(results) -> Dict[str, Figure] 全局可视化
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import visualizer
from .absa_classifier import ABSASentimentClassifier
from .aspect_extractor import AspectExtractor, lexicon_extract
from .explainer import LIMEExplainer, OcclusionExplainer
from .keyword_extractor import KeywordExtractor
from .llm_utils import get_default_client
from .preprocessor import TextPreprocessor
from .sentiment_classifier import SentimentClassifier
from .summarizer import OpinionSummarizer


class ReviewAnalyzerApp:
    """影评智能分析系统主程序（模块集成层）。"""

    def __init__(self, project_dir: str = "movie_review_absa", env_path: Optional[str] = None,
                 model_name: str = "hfl/chinese-bert-wwm-ext",
                 use_llm_summary: bool = False, device=None, verbose: bool = True):
        self.project_dir = project_dir
        self.data_dir = os.path.join(project_dir, "data")
        self.model_dir = os.path.join(project_dir, "models")
        self.verbose = verbose

        self.preprocessor = TextPreprocessor(
            os.path.join(self.data_dir, "stopwords.txt"),
            os.path.join(self.data_dir, "user_dict.txt"),
        )
        self.keyword_extractor = KeywordExtractor(
            os.path.join(self.data_dir, "stopwords.txt"),
            os.path.join(self.data_dir, "user_dict.txt"),
        )
        env_path = env_path or os.path.join(os.path.dirname(os.path.abspath(project_dir)), ".env")
        self.llm_client = get_default_client(env_path) if use_llm_summary else None
        self.summarizer = OpinionSummarizer(llm_client=self.llm_client)
        self.lime = LIMEExplainer()
        self.occlusion = OcclusionExplainer()

        # ---------------- 方面抽取器 ----------------
        self.aspect_extractor = self._load_aspect_extractor(device)
        # ---------------- 情感分类器 ----------------
        self.overall_clf = self._load_cls(
            SentimentClassifier, os.path.join(self.model_dir, "sentiment_classifier"), model_name, device)
        self.absa_clf = self._load_cls(
            ABSASentimentClassifier, os.path.join(self.model_dir, "absa_classifier"), model_name, device)

    def _load_aspect_extractor(self, device):
        candidates = [
            os.path.join(self.model_dir, "aspect_extractor", "bert_crf"),
            os.path.join(self.model_dir, "aspect_extractor"),
        ]
        for path in candidates:
            if os.path.exists(os.path.join(path, "model.pt")):
                try:
                    obj = AspectExtractor.load(path, device=device)
                    if self.verbose:
                        print(f"已加载方面抽取模型：{path}")
                    return obj
                except Exception as exc:
                    print(f"加载 {path} 失败：{exc}")
        bilstm_path = os.path.join(self.model_dir, "aspect_extractor", "bilstm_crf")
        if os.path.exists(os.path.join(bilstm_path, "bilstm_crf.pt")):
            obj = AspectExtractor.load(bilstm_path, device=device)
            if self.verbose:
                print("已加载 BiLSTM-CRF 方面抽取模型")
            return obj
        print("[提示] 未找到方面抽取模型，将使用词典回退方案")
        return None

    def _load_cls(self, cls, path: str, model_name: str, device):
        if os.path.exists(os.path.join(path, "config.json")):
            try:
                obj = cls.load(path, device=device)
                if self.verbose:
                    print(f"已加载 {cls.__name__}：{path}")
                return obj
            except Exception as exc:
                print(f"加载 {path} 失败：{exc}")
        if self.verbose:
            print(f"[提示] 未找到 {path}，使用未微调的预训练模型（demo 模式）")
        return cls(model_name=model_name, device=device)

    # ------------------------------------------------------------------
    def analyze(self, text: str, top_k: int = 5, with_explanation: bool = False) -> Dict:
        """文档接口 analyze(text) -> ReviewResult。"""
        pre = self.preprocessor.preprocess(text)
        clean = pre["clean_text"]

        aspects: List[Dict] = []
        if self.aspect_extractor is not None:
            aspects = self.aspect_extractor.extract(clean, use_lexicon_fallback=True)
        if not aspects:
            aspects = lexicon_extract(clean)

        for asp in aspects:
            if self.absa_clf is not None:
                senti, conf = self.absa_clf.predict(clean or text, asp["aspect_term"])
                asp["sentiment"], asp["confidence"] = senti, conf

        if self.overall_clf is not None:
            overall, overall_conf = self.overall_clf.predict(clean or text)
            dist = self.overall_clf.predict_proba(clean or text)
        else:  # 极简回退：用方面情感投票
            votes = [a.get("sentiment") for a in aspects if a.get("sentiment")]
            overall = max(set(votes), key=votes.count) if votes else "neutral"
            overall_conf, dist = 0.0, [0.0, 0.0, 0.0]

        keywords = self.keyword_extractor.extract(clean or text, top_k=top_k)
        summary = self.summarizer.summarize(text, aspects)
        explanation: Dict = {}
        if with_explanation and self.overall_clf is not None:
            explanation = self.lime.explain(clean or text, self.overall_clf)

        return {
            "raw_text": text,
            "clean_text": clean,
            "overall_sentiment": overall,
            "overall_confidence": round(float(overall_conf), 4),
            "overall_distribution": {k: round(float(v), 4)
                                     for k, v in zip(["negative", "neutral", "positive"], dist)},
            "aspect_sentiments": [
                {
                    "aspect_term": a["aspect_term"],
                    "aspect_category": a["aspect_category"],
                    "sentiment": a.get("sentiment"),
                    "confidence": round(float(a.get("confidence") or 0.0), 4),
                    "source": a.get("source", "model"),
                } for a in aspects
            ],
            "keywords": keywords,
            "summary": summary,
            "explanation": explanation,
            "flags": {"negation": pre["negation"], "irony": pre["irony"]},
        }

    # ------------------------------------------------------------------
    def analyze_batch(self, file_path: str, text_column: str = "text",
                      output_path: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """文档接口 analyze_batch(file_path) -> List[ReviewResult]。"""
        if file_path.endswith(".jsonl"):
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            texts = [r.get(text_column) or r.get("raw_text") or "" for r in records]
        else:
            df = pd.read_csv(file_path)
            texts = df[text_column].astype(str).tolist()
        if limit:
            texts = texts[:limit]
        results = [self.analyze(t) for t in texts if t and str(t).strip()]
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            if output_path.endswith(".csv"):
                pd.DataFrame([{
                    "raw_text": r["raw_text"],
                    "overall_sentiment": r["overall_sentiment"],
                    "overall_confidence": r["overall_confidence"],
                    "aspects": ";".join(f"{a['aspect_term']}({a['sentiment']})"
                                        for a in r["aspect_sentiments"]),
                    "keywords": ",".join(r["keywords"]),
                    "summary": r["summary"],
                } for r in results]).to_csv(output_path, index=False, encoding="utf-8-sig")
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return results

    # ------------------------------------------------------------------
    def explain(self, text: str, method: str = "lime") -> Dict:
        clean = self.preprocessor.clean(text)
        engine = self.lime if method == "lime" else self.occlusion
        return engine.explain(clean, self.overall_clf)

    def visualize(self, results: Sequence[Dict]) -> Dict[str, object]:
        """文档接口 visualize(results)：情感分布 / 方面热力图 / 高频方面 / 关键词。"""
        counts = {"negative": 0, "neutral": 0, "positive": 0}
        aspect_flat, texts = [], []
        for r in results:
            senti = r.get("overall_sentiment", "neutral")
            counts[senti] = counts.get(senti, 0) + 1
            aspect_flat.extend(r.get("aspect_sentiments", []))
            texts.append(r.get("clean_text") or r.get("raw_text", ""))
        figs = {
            "sentiment_pie": visualizer.plot_sentiment_pie(counts),
            "aspect_heatmap": visualizer.plot_aspect_heatmap(aspect_flat),
            "aspect_frequency": visualizer.plot_aspect_frequency(aspect_flat),
        }
        if texts:
            pairs = self.keyword_extractor.global_keywords_with_weights(texts, top_k=15)
            figs["keywords"] = visualizer.plot_keyword_barh(pairs, title="语料关键词（TF-IDF 全局）")
            figs["keyword_bubbles"] = visualizer.plot_word_frequency_bubbles(pairs)
        return figs
