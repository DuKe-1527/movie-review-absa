# -*- coding: utf-8 -*-
"""
批量分析脚本（工程规范）

用法（工作目录 E:\\AIagent\\DB1，DB1 环境）：
    python movie_review_absa/analyze.py --text "特效很震撼，但剧情拖沓"
    python movie_review_absa/analyze.py --input movie_review_absa/data/reviews_raw_unlabeled.jsonl ^
           --output movie_review_absa/outputs/batch_results.csv --limit 20 --explain
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyzer import ReviewAnalyzerApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="movie_review_absa")
    parser.add_argument("--text", default=None, help="单条评论分析")
    parser.add_argument("--input", default=None, help="批量输入文件（csv / jsonl）")
    parser.add_argument("--output", default=None, help="批量结果输出路径（csv / jsonl）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--explain", action="store_true", help="单条分析时输出 LIME 解释")
    parser.add_argument("--llm-summary", action="store_true", help="使用大模型生成观点摘要")
    args = parser.parse_args()

    app = ReviewAnalyzerApp(project_dir=args.project_dir, use_llm_summary=args.llm_summary)

    if args.text:
        result = app.analyze(args.text, with_explanation=args.explain)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.input:
        output = args.output or os.path.join(args.project_dir, "outputs", "batch_results.csv")
        results = app.analyze_batch(args.input, output_path=output, limit=args.limit)
        print(f"共分析 {len(results)} 条，结果已保存：{output}")
        for r in results[:3]:
            print(f"- {r['raw_text'][:28]}… → {r['overall_sentiment']}"
                  f"｜方面 {len(r['aspect_sentiments'])} 个｜{r['summary'][:40]}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
