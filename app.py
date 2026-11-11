# -*- coding: utf-8 -*-
"""
任务七：影评智能分析 Web 界面（Streamlit）

运行（工作目录 E:\\AIagent\\DB1，DB1 环境）：
    streamlit run movie_review_absa/app.py

三个功能区：
    1) 单条评论分析：整体情感 + 方面级情感 + 关键词 + 观点摘要 + LIME 可解释性
    2) 批量文件分析：上传 CSV/JSONL，下载分析结果
    3) 全局统计：情感分布、方面-情感热力图、高频方面、关键词分布
"""
from __future__ import annotations

import io
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyzer import ReviewAnalyzerApp
from src.explainer import plot_feature_importance
from src.visualizer import set_chinese_font

st.set_page_config(page_title="影评方面级情感分析与观点摘要", layout="wide")
set_chinese_font()

SAMPLE = "这部电影的特效非常震撼，画面极具想象力，配乐也相当加分；但是剧情拖沓，结尾让人失望，演员的表演略显浮夸。"


@st.cache_resource(show_spinner="正在加载模型（首次加载约需 10~30 秒）…")
def get_app(project_dir: str) -> ReviewAnalyzerApp:
    return ReviewAnalyzerApp(project_dir=project_dir, verbose=False)


def render_result(result: dict, analyzer: ReviewAnalyzerApp) -> None:
    zh = {"negative": "负面", "neutral": "中性", "positive": "正面"}
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("整体情感", zh.get(result["overall_sentiment"], result["overall_sentiment"]),
                  f"置信度 {result['overall_confidence']:.2f}")
    with col2:
        dist = pd.DataFrame.from_dict(result["overall_distribution"], orient="index",
                                      columns=["概率"])
        st.write("**情感分布**")
        st.bar_chart(dist)

    st.subheader("方面级情感")
    if result["aspect_sentiments"]:
        st.dataframe(pd.DataFrame([{
            "方面术语": a["aspect_term"],
            "方面类别": a["aspect_category"],
            "情感": zh.get(a["sentiment"], a["sentiment"]),
            "置信度": a["confidence"],
            "来源": a["source"],
        } for a in result["aspect_sentiments"]]), use_container_width=True, hide_index=True)
    else:
        st.info("未识别到显式方面词（该评论可能只表达整体感受）。")

    st.subheader("关键词与观点摘要")
    st.write("关键词：" + "、".join(result["keywords"]))
    st.success(result["summary"])

    if result["flags"]["irony"] or result["flags"]["negation"]:
        st.warning(f"检测到特殊表达：反讽={result['flags']['irony']}，否定={result['flags']['negation']}"
                   "（此类样本模型易出错，见任务六错误分析）")

    with st.expander("模型可解释性（LIME 词级贡献）", expanded=False):
        if st.button("生成 LIME 解释", key="lime_btn"):
            with st.spinner("正在计算 LIME 特征贡献…"):
                exp = analyzer.explain(result["raw_text"], method="lime")
            st.pyplot(plot_feature_importance(exp))
            st.caption("绿色为支持当前预测的词，红色为反对当前预测的词。")


def main() -> None:
    st.title("🎬 影评方面级情感分析与观点摘要系统")
    st.caption("流程：文本预处理 → 方面术语抽取(BERT-CRF) → 方面级情感(句对 BERT) → 关键词 → 观点摘要 → LIME 解释")

    project_dir = st.sidebar.text_input("项目目录", "movie_review_absa")
    analyzer = get_app(project_dir)

    tab1, tab2, tab3 = st.tabs(["单条评论分析", "批量文件分析", "全局统计"])

    with tab1:
        text = st.text_area("输入电影评论", SAMPLE, height=120)
        if st.button("开始分析", type="primary"):
            with st.spinner("分析中…"):
                st.session_state["last_result"] = analyzer.analyze(text)
        if "last_result" in st.session_state:
            render_result(st.session_state["last_result"], analyzer)

    with tab2:
        uploaded = st.file_uploader("上传 CSV（含 text 列）或 JSONL（含 text/raw_text 字段）",
                                    type=["csv", "jsonl"])
        limit = st.number_input("最多分析条数", min_value=1, max_value=2000, value=50, step=10)
        if uploaded is not None and st.button("批量分析"):
            raw = uploaded.read()
            if uploaded.name.endswith(".csv"):
                texts = pd.read_csv(io.BytesIO(raw))["text"].astype(str).tolist()
            else:
                texts = []
                for line in raw.decode("utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        texts.append(rec.get("text") or rec.get("raw_text", ""))
            texts = [t for t in texts if t.strip()][: int(limit)]
            bar = st.progress(0.0)
            results = []
            for i, t in enumerate(texts):
                results.append(analyzer.analyze(t))
                bar.progress((i + 1) / len(texts))
            st.session_state["batch_results"] = results
            df = pd.DataFrame([{
                "评论": r["raw_text"],
                "整体情感": r["overall_sentiment"],
                "置信度": r["overall_confidence"],
                "方面结果": "; ".join(f"{a['aspect_term']}({a['sentiment']})"
                                      for a in r["aspect_sentiments"]),
                "关键词": ",".join(r["keywords"]),
                "摘要": r["summary"],
            } for r in results])
            st.dataframe(df, use_container_width=True)
            st.download_button("下载分析结果 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="absa_results.csv", mime="text/csv")

    with tab3:
        source = st.radio("统计来源", ["内置示例语料", "使用批量分析结果"], horizontal=True)
        results = None
        if source == "使用批量分析结果" and "batch_results" in st.session_state:
            results = st.session_state["batch_results"]
        if results is None:
            results = []
            sample_path = os.path.join(project_dir, "data", "reviews_absa.jsonl")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 40:
                            break
                        rec = json.loads(line)
                        results.append({
                            "raw_text": rec["text"],
                            "clean_text": rec["text"],
                            "overall_sentiment": rec.get("overall_sentiment", "neutral"),
                            "aspect_sentiments": rec.get("aspects", []),
                        })
            if not results:
                st.info("暂无统计数据，请先在“批量文件分析”中上传文件。")
                return
        figs = analyzer.visualize(results)
        c1, c2 = st.columns(2)
        c1.pyplot(figs["sentiment_pie"])
        c2.pyplot(figs["aspect_heatmap"])
        st.pyplot(figs["aspect_frequency"])
        if "keywords" in figs:
            st.pyplot(figs["keywords"])


if __name__ == "__main__":
    main()
