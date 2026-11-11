# -*- coding: utf-8 -*-
"""
任务二：中文文本深度预处理（文档 6.1 TextPreprocessor）

接口（与任务书一致）：
    clean(text) -> str
    tokenize(text) -> List[Tuple[str, str]]        # (词语, 词性)
    remove_stopwords(tokens) -> List[Tuple[str, str]]

扩展能力（面向影评场景）：
    normalize(text)      网络用语 / 表情符号 / 重复字符 归一化
    preprocess(text)     一次性给出清洗、分词、词性、反讽与否定标记
    is_irony(text)       反讽表达启发式判定
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import jieba
import jieba.posseg as pseg

# --------------------------------------------------------------------------
# 1. 影评场景资源：表情符号、网络用语、情感词、否定词、反讽线索
# --------------------------------------------------------------------------
EMOTICON_MAP: Dict[str, str] = {
    "😂": "笑哭", "🤣": "爆笑", "😭": "大哭", "😅": "尴尬笑", "😊": "微笑",
    "😍": "喜欢", "😴": "无聊", "😡": "愤怒", "👍": "点赞", "👎": "差评",
    "🤔": "思考", "😱": "震惊", "🔥": "火爆", "💩": "糟糕", "🙄": "无语",
    "(*^_^*)": "微笑", "^_^": "微笑", "T_T": "大哭", ":)": "微笑",
}

NET_SLANG: Dict[str, str] = {
    "yyds": "永远的神", "YYDS": "永远的神", "xswl": "笑死我了", "awsl": "啊我死了",
    "nbcs": "没人在乎", "u1s1": "有一说一", "srds": "虽然但是", "zqsg": "真情实感",
    "kdl": "太好嗑了", "emmm": "犹豫", "2333": "大笑", "233": "大笑",
    "666": "厉害", "hhh": "大笑", "笑死": "好笑", "泪目": "感动",
    "绝绝子": "非常棒", "拉胯": "很差", "踩雷": "体验很差", "封神": "非常优秀",
    "无语子": "无语", "栓Q": "谢谢", "真的会谢": "无奈",
}

# 情感词：1) 分词时保护情感词不被停用词过滤；2) 无 API 时弱标注回退
POSITIVE_WORDS = set("""
好 很好 好看 精彩 优秀 出色 惊艳 震撼 经典 完美 喜欢 爱 感动 泪目 温暖 治愈 舒服 过瘾 燃 有趣 幽默 搞笑
细腻 深刻 真实 真诚 用心 高级 顶级 一流 神作 佳作 力荐 推荐 值得 不虚此行 意犹未尽 拍案叫绝 赞 赞美 惊叹
漂亮 唯美 精致 宏大 壮丽 震撼人心 引人入胜 出彩 亮眼 加分 惊喜 满足 幸福 贴心 流畅 丝滑 自然 到位 在线 稳
""".split())

NEGATIVE_WORDS = set("""
差 很差 难看 糟糕 垃圾 无聊 拖沓 冗长 尴尬 出戏 崩 崩坏 失望 后悔 空洞 苍白 生硬 牵强 刻意 做作 矫情
烂 烂片 平庸 敷衍 圈钱 油腻 混乱 迷惑 无趣 煎熬 受不了 恶心 难看死 浪费时间 不值 劝退 避雷
拉胯 割裂 断片 浮夸 呆板 死板 浅薄 幼稚 中二 尬 尬演 面瘫 端水 注水 拖 离谱 迷惑行为 强行 硬凑 虎头蛇尾 烂尾
""".split())

SENTIMENT_LEXICON = POSITIVE_WORDS | NEGATIVE_WORDS

NEGATION_WORDS = {
    "不", "没", "没有", "无", "未", "别", "非", "毫无", "谈不上", "算不上",
    "不能", "不会", "不要", "不太", "不够", "称不上", "并非",
}

# 反讽 / 反话线索词（任务六错误分析会用到）
IRONY_CUES = [
    "呵呵", "就这", "笑死", "笑而不语", "厉害了", "真厉害", "真好呢", "有意思",
    "谢谢啊", "谢谢了", "勇气可嘉", "佩服", "绝了", "满分", "最佳",
    "居然", "果然", "服了", "无语", "无力吐槽", "忍俊不禁",
]

_REPEAT_RE = re.compile(r"(.)\1{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_AT_TOPIC_RE = re.compile(r"[@#]\S+")
_KEEP_RE = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；“”‘’（）《》…—,.!?;:'\"()\[\]%~\-]")


class TextPreprocessor:
    """中文影评文本清洗、归一化、分词与词性标注。"""

    def __init__(
        self,
        stopwords_path: str,
        user_dict_path: Optional[str] = None,
        keep_sentiment_words: bool = True,
    ) -> None:
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as f:
                self.stopwords = {line.strip() for line in f if line.strip()}
        self.keep_sentiment_words = keep_sentiment_words
        self.user_dict_path = None
        if user_dict_path and os.path.exists(user_dict_path):
            jieba.load_userdict(user_dict_path)
            self.user_dict_path = user_dict_path
            with open(user_dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if parts and len(parts[0]) > 1:
                        jieba.add_word(parts[0])

    # ------------------------- 归一化 -------------------------
    def normalize(self, text: str) -> str:
        """表情符号 → 文字、网络用语 → 规范表达、重复字符压缩。"""
        text = str(text)
        for k, v in EMOTICON_MAP.items():
            text = text.replace(k, v)
        for k, v in NET_SLANG.items():
            text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
        text = _REPEAT_RE.sub(r"\1\1", text)  # “好好好好” → “好好”
        return text

    # ------------------------- 清洗 -------------------------
    def clean(self, text: str) -> str:
        """文档接口 clean(text) -> str：去链接 / @话题 / 非法字符，压缩空白。"""
        text = self.normalize(text)
        text = _URL_RE.sub(" ", text)
        text = _AT_TOPIC_RE.sub(" ", text)
        text = _KEEP_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------- 分词 -------------------------
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """文档接口 tokenize(text) -> [(词语, 词性)]，已去停用词。"""
        tokens: List[Tuple[str, str]] = []
        for word, flag in pseg.cut(self.normalize(text)):
            w = word.strip()
            if not w:
                continue
            if w in self.stopwords and not (self.keep_sentiment_words and w in SENTIMENT_LEXICON):
                continue
            if all(not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fa5") for ch in w):
                continue  # 纯标点
            tokens.append((w, flag))
        return tokens

    def tokenize_raw(self, text: str) -> List[Tuple[str, str]]:
        """不做停用词过滤的分词结果（用于效果对比展示）。"""
        return [(w, f) for w, f in pseg.cut(self.normalize(text)) if w.strip()]

    def remove_stopwords(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """文档接口 remove_stopwords(tokens)。"""
        out = []
        for w, pos in tokens:
            if w in self.stopwords and not (self.keep_sentiment_words and w in SENTIMENT_LEXICON):
                continue
            out.append((w, pos))
        return out

    # ------------------------- 进阶工具 -------------------------
    def words(self, text: str) -> List[str]:
        return [w for w, _ in self.tokenize(text)]

    def has_negation(self, text: str) -> bool:
        return any(neg in text for neg in NEGATION_WORDS)

    def is_irony(self, text: str) -> bool:
        """反讽启发式：反讽线索词 + 引号/波浪号等组合信号。"""
        score = sum(1 for cue in IRONY_CUES if cue in text)
        if any(p in text for p in ("呵呵", "就这", "笑死", "无语")):
            score += 1
        if re.search(r"[“”\"]", text) and re.search(r"(真|好|棒|优秀|经典|满分)", text):
            score += 1
        if re.search(r"(～|~){1,}", text):
            score += 1
        return score >= 2

    def preprocess(self, text: str) -> Dict:
        """一次性输出预处理结果，供下游模块（抽取 / 分类 / 摘要）使用。"""
        clean_text = self.clean(text)
        tokens = self.tokenize(clean_text)
        return {
            "raw_text": text,
            "clean_text": clean_text,
            "tokens": tokens,
            "words": [w for w, _ in tokens],
            "pos": [p for _, p in tokens],
            "negation": self.has_negation(clean_text),
            "irony": self.is_irony(clean_text),
        }
