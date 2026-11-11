# 影评方面级情感分析与观点摘要系统

基于中文电影短评的 ABSA（Aspect-Based Sentiment Analysis）综合项目，覆盖数据构建、方面抽取、情感分类、观点摘要、可解释性与可视化界面。

## 核心能力

| 模块 | 实现 |
| --- | --- |
| 整体情感分类 | 中文 BERT 微调 + TF-IDF/SVM 基线 |
| 方面术语抽取 | BERT-CRF、BiLSTM-CRF |
| 方面级情感分类 | `[CLS] 评论 [SEP] 方面 [SEP]` 句对 BERT |
| 关键词抽取 | TF-IDF、TextRank、RAKE 融合 |
| 观点摘要 | 抽取式摘要，可选 LLM 生成式摘要 |
| 可解释性 | LIME 词级贡献、逐词遮挡与错误案例 |
| 可视化 | Streamlit 单条分析、批量分析与全局图表 |

## 目录

```text
app.py                 Streamlit 交互界面
train.py               模型训练入口
analyze.py             单条 / 批量分析入口
src/                   预处理、模型、解释与可视化模块
data/                  处理后数据、标注规范与样例数据
metrics/               模型评测指标
outputs/               示例分析结果与可视化图表
notebooks/             完整实验 Notebook
```

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows 可使用：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 使用

```bash
python train.py --epochs 2 --max-samples 800
python analyze.py --text "特效很震撼，但剧情拖沓"
python analyze.py --input data/reviews_raw_unlabeled.jsonl --output outputs/batch.csv --limit 20
streamlit run app.py
```

## 数据与模型说明

- 项目使用公开豆瓣短评数据集，经清洗后抽取 3000+ 条样本。
- 方面级标注采用“大模型辅助标注 + 规则校验 + 人工抽检”。
- 仓库不包含原始 57 MB 缓存、BERT 权重和训练中间文件，避免仓库体积过大；可按 `train.py` 重新训练。
- 模型评测结果保存在 `metrics/train_metrics.json`。
- 可选的 DeepSeek / 智谱 API 通过环境变量 `DEEPSEEK_API_KEY`、`ZHIPUAI_API_KEY` 读取，不写入源码。

## 技术栈

Python · PyTorch · Transformers · BERT-CRF · BiLSTM-CRF · scikit-learn · jieba · LIME · Streamlit
