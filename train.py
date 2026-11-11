# -*- coding: utf-8 -*-
"""
任务三 / 任务四 训练脚本（工程规范）

用法（工作目录 E:\\AIagent\\DB1，DB1 环境）：
    python movie_review_absa/train.py --epochs 2 --max-samples 800
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.absa_classifier import ABSASentimentClassifier
from src.aspect_extractor import AspectExtractor
from src.preprocessor import TextPreprocessor
from src.sentiment_classifier import SentimentClassifier, TFIDFSVMBaseline, LABELS


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_dataset(records, ratios=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    data = records[:]
    random.shuffle(data)
    n_train = int(len(data) * ratios[0])
    n_val = int(len(data) * ratios[1])
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="movie_review_absa")
    parser.add_argument("--model-name", default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_path = os.path.join(args.project_dir, "data", "reviews_absa.jsonl")
    records = load_jsonl(data_path)[: args.max_samples]
    train, val, test = split_dataset(records)
    print(f"数据划分：train={len(train)} val={len(val)} test={len(test)}")

    metrics = {}
    # ---------------- 任务三：方面术语抽取 ----------------
    extractor = AspectExtractor(model_type="bert", model_name=args.model_name, max_len=64)
    extractor.train([{"text": r["text"], "aspects": r["aspects"]} for r in train],
                    epochs=args.epochs, batch_size=args.batch_size)
    metrics["aspect_extractor"] = extractor.evaluate(
        [{"text": r["text"], "aspects": r["aspects"]} for r in test])
    extractor.save(os.path.join(args.project_dir, "models", "aspect_extractor", "bert_crf"))

    # ---------------- 任务四：整体情感 ----------------
    texts = [r["text"] for r in train]
    labels = [r["overall_sentiment"] for r in train]
    overall = SentimentClassifier(model_name=args.model_name, max_len=64)
    overall.train(texts, labels, epochs=args.epochs, batch_size=args.batch_size)
    metrics["overall_sentiment"] = overall.evaluate([r["text"] for r in test],
                                                    [r["overall_sentiment"] for r in test])
    overall.save(os.path.join(args.project_dir, "models", "sentiment_classifier"))

    pre = TextPreprocessor(os.path.join(args.project_dir, "data", "stopwords.txt"),
                           os.path.join(args.project_dir, "data", "user_dict.txt"))
    baseline = TFIDFSVMBaseline(tokenizer_fn=pre.words)
    baseline.fit(texts, labels)
    metrics["overall_sentiment_tfidf_svm"] = baseline.evaluate(
        [r["text"] for r in test], [r["overall_sentiment"] for r in test])
    baseline.save(os.path.join(args.project_dir, "models", "sentiment_classifier", "tfidf_svm.pkl"))

    # ---------------- 任务四：方面级情感 ----------------
    a_texts, a_aspects, a_labels = [], [], []
    for r in train:
        for asp in r["aspects"]:
            if asp.get("sentiment") in LABELS:
                a_texts.append(r["text"])
                a_aspects.append(asp["term"])
                a_labels.append(asp["sentiment"])
    t_texts, t_aspects, t_labels = [], [], []
    for r in test:
        for asp in r["aspects"]:
            if asp.get("sentiment") in LABELS:
                t_texts.append(r["text"])
                t_aspects.append(asp["term"])
                t_labels.append(asp["sentiment"])
    absa = ABSASentimentClassifier(model_name=args.model_name, max_len=96)
    absa.train(a_texts, a_aspects, a_labels, epochs=args.epochs, batch_size=args.batch_size)
    metrics["aspect_sentiment"] = absa.evaluate(t_texts, t_aspects, t_labels)
    absa.save(os.path.join(args.project_dir, "models", "absa_classifier"))

    out = os.path.join(args.project_dir, "models", "train_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("训练完成，指标已保存：", out)


if __name__ == "__main__":
    main()
