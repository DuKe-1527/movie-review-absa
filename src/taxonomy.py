# -*- coding: utf-8 -*-
"""
方面类别体系与电影领域方面词典（任务一“方面类别设计”的统一来源）

被以下模块共用，保证标注、抽取、可视化使用同一套类别定义：
    src/aspect_extractor.py   方面术语抽取（类别映射 / 词典回退）
    src/llm_utils.py          大模型辅助标注的类别约束
    src/visualizer.py         方面-情感热力图的类别顺序
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# 8 个业务方面类别（可在项目评审时按业务需求调整）
ASPECT_CATEGORY_LIST: List[str] = [
    "剧本剧情", "演员表演", "视觉效果", "导演水平",
    "音乐音效", "叙事节奏", "台词文本", "整体体验",
]
UNKNOWN_CATEGORY = "其他"

# 电影领域方面词典（term -> category）
ASPECT_LEXICON: Dict[str, str] = {
    # 剧本剧情
    "剧情": "剧本剧情", "故事": "剧本剧情", "剧本": "剧本剧情", "情节": "剧本剧情",
    "结尾": "剧本剧情", "结局": "剧本剧情", "剧情线": "剧本剧情", "设定": "剧本剧情",
    "主题": "剧本剧情", "内涵": "剧本剧情", "人物": "剧本剧情", "角色": "剧本剧情",
    "主角": "剧本剧情", "配角": "剧本剧情", "反派": "剧本剧情", "故事线": "剧本剧情",
    "桥段": "剧本剧情", "转折": "剧本剧情", "伏笔": "剧本剧情", "逻辑": "剧本剧情",
    "立意": "剧本剧情", "世界观": "剧本剧情", "原作": "剧本剧情", "改编": "剧本剧情",
    # 演员表演
    "演员": "演员表演", "演技": "演员表演", "表演": "演员表演", "主演": "演员表演",
    "女主": "演员表演", "男主": "演员表演", "群演": "演员表演", "表情": "演员表演",
    "选角": "演员表演", "对手戏": "演员表演", "哭戏": "演员表演", "感情戏": "演员表演",
    # 视觉效果
    "特效": "视觉效果", "画面": "视觉效果", "镜头": "视觉效果", "摄影": "视觉效果",
    "美术": "视觉效果", "布景": "视觉效果", "色彩": "视觉效果", "色调": "视觉效果",
    "场景": "视觉效果", "视效": "视觉效果", "CG": "视觉效果", "3D": "视觉效果",
    "IMAX": "视觉效果", "构图": "视觉效果", "打光": "视觉效果", "服化道": "视觉效果",
    "道具": "视觉效果", "剪辑": "视觉效果", "动作戏": "视觉效果", "打斗": "视觉效果",
    "长镜头": "视觉效果", "运镜": "视觉效果",
    # 导演水平
    "导演": "导演水平", "执导": "导演水平", "调度": "导演水平", "掌控": "导演水平",
    "叙事手法": "导演水平", "风格": "导演水平", "作者性": "导演水平", "镜头语言": "导演水平",
    "场面调度": "导演水平",
    # 音乐音效
    "配乐": "音乐音效", "音乐": "音乐音效", "音效": "音乐音效", "插曲": "音乐音效",
    "主题曲": "音乐音效", "BGM": "音乐音效", "bgm": "音乐音效", "声音": "音乐音效",
    "歌曲": "音乐音效", "配音": "音乐音效", "音响": "音乐音效", "片尾曲": "音乐音效",
    # 叙事节奏
    "节奏": "叙事节奏", "叙事": "叙事节奏", "铺垫": "叙事节奏", "高潮": "叙事节奏",
    "时长": "叙事节奏", "分段": "叙事节奏", "结构": "叙事节奏", "非线性": "叙事节奏",
    # 台词文本
    "台词": "台词文本", "对白": "台词文本", "旁白": "台词文本", "金句": "台词文本",
    "字幕": "台词文本", "文本": "台词文本", "独白": "台词文本",
    # 整体体验
    "电影": "整体体验", "片子": "整体体验", "影片": "整体体验", "本片": "整体体验",
    "这部电影": "整体体验", "观感": "整体体验", "观影体验": "整体体验",
    "整体": "整体体验", "氛围": "整体体验", "感受": "整体体验", "笑点": "整体体验",
    "泪点": "整体体验", "情怀": "整体体验", "爽感": "整体体验",
}


def aspect_category(term: str) -> str:
    """方面术语 → 方面类别：先精确匹配，再最长子串匹配，最后返回“其他”。"""
    term = (term or "").strip()
    if not term:
        return UNKNOWN_CATEGORY
    if term in ASPECT_LEXICON:
        return ASPECT_LEXICON[term]
    best, best_len = UNKNOWN_CATEGORY, 0
    for key, cat in ASPECT_LEXICON.items():
        if key in term and len(key) > best_len:
            best, best_len = cat, len(key)
    for key, cat in ASPECT_LEXICON.items():
        if term in key and len(term) >= 2 and len(term) > best_len:
            best, best_len = cat, len(term)
    return best


def lexicon_extract(text: str) -> List[Dict]:
    """词典回退方案：模型输出为空或服务降级时使用。"""
    hits: List[Tuple[int, int, str]] = []
    for term in ASPECT_LEXICON:
        start = text.find(term)
        while start != -1:
            hits.append((start, start + len(term), term))
            start = text.find(term, start + 1)
    hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: List[Tuple[int, int, str]] = []
    for s, e, t in hits:  # 去掉被更长命中完全包含的短命中
        if any(s >= ks and e <= ke for ks, ke, _ in kept):
            continue
        kept.append((s, e, t))
    out, seen = [], set()
    for _, _, term in kept:
        if term in seen:
            continue
        seen.add(term)
        out.append({
            "aspect_term": term,
            "aspect_category": aspect_category(term),
            "sentiment": None,
            "confidence": 0.0,
            "source": "lexicon",
        })
    return out
