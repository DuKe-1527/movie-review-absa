# -*- coding: utf-8 -*-
"""
大模型工具模块（工程扩展）

用途：
    1) 任务一：批量辅助标注方面术语 / 类别 / 极性（LLM 作为标注员，人工抽检复核）
    2) 任务五：生成式观点摘要（文档“进阶：BART / T5 / ChatGLM 生成式摘要”）

说明：
    - API Key 从工作区 .env 读取（DEEPSEEK_API_KEY / ZHIPUAI_API_KEY）；
    - 全部调用都带重试与超时；不可用时返回 None，调用方自动回退到规则方案。
"""
from __future__ import annotations

import json
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

import requests

from .taxonomy import ASPECT_CATEGORY_LIST, aspect_category

SENTIMENT_LABELS = ["positive", "neutral", "negative"]


def load_env_file(path: str) -> Dict[str, str]:
    """读取 .env 文件（只做解析，不回显密钥内容）。"""
    env: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class LLMClient:
    """极简 OpenAI 兼容 Chat Completions 客户端。"""

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat",
                 timeout: int = 120, max_retry: int = 3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retry = max_retry

    def chat(self, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 2000,
             json_mode: bool = False) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(self.max_retry):
            try:
                resp = requests.post(
                    self.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.api_key,
                             "Content-Type": "application/json"},
                    json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:  # 网络异常 / 超时
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
        print(f"    [LLM] 调用失败（已重试 {self.max_retry} 次）：{last_err}")
        return None


def get_default_client(env_path: str, model: str = "deepseek-chat") -> Optional[LLMClient]:
    """按优先级构造可用客户端：DeepSeek → 智谱 GLM。"""
    env = load_env_file(env_path)
    if env.get("DEEPSEEK_API_KEY"):
        return LLMClient(env["DEEPSEEK_API_KEY"],
                         env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), model=model)
    if env.get("ZHIPUAI_API_KEY"):
        return LLMClient(env["ZHIPUAI_API_KEY"],
                         env.get("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
                         model=model if model.startswith("glm") else "glm-4-flash")
    return None


def parse_json_array(text: str) -> Optional[List]:
    """从模型输出中稳健地解析 JSON 数组。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.lstrip("json").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("data", "items", "results", "annotations"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        return data if isinstance(data, list) else None
    except Exception:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------
# 任务一：大模型辅助方面级标注（LLM as annotator）
# --------------------------------------------------------------------------
ANNOTATION_SYSTEM = "你是中文电影短评的方面级情感分析标注专家，严格按照标注规范输出 JSON。"


def build_annotation_prompt(texts: Sequence[str], categories: Sequence[str]) -> str:
    """构造批量标注提示：一次请求标注多条短评（批内独立、互不影响）。"""
    cat_str = "、".join(categories)
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        "请对下面的电影短评逐条做方面级情感标注。\n"
        f"方面类别体系（必须从中选择）：{cat_str}\n"
        "标注规范：\n"
        "1) 只抽取评论中【显式出现】的方面术语，术语必须是原文中连续出现的词语"
        "（例如“特效”“剧情”“演员的表演”“配乐”），不要抽取情感词（如“震撼”“失望”）；\n"
        "2) 每个方面给出类别与情感极性：positive / negative / neutral；\n"
        "3) 若评论只表达整体感受、没有显式方面词，aspects 返回空列表 [];\n"
        "4) 整体情感 polarity 用 positive / negative / neutral 表示；\n"
        "5) 严格输出 JSON 对象，形如 "
        '{"annotations": [{"id": 1, "aspects": [{"term": "剧情", "category": "剧本剧情", '
        '"sentiment": "negative"}], "overall": "negative"}]}，不要输出任何解释。\n'
        f"待标注评论：\n{lines}"
    )


def _clean_annotation(item: Dict, text: str, categories: Sequence[str]) -> Dict:
    """规则校验：术语必须出现在原文、类别必须合法、极性必须合法。"""
    aspects = []
    seen = set()
    for asp in (item.get("aspects") or []):
        if not isinstance(asp, dict):
            continue
        term = str(asp.get("term", "")).strip()
        if not term or term not in text or term in seen:
            continue
        category = asp.get("category")
        if category not in categories:
            category = aspect_category(term)
            if category == "其他":
                category = "整体体验" if len(term) >= 3 else None
        sentiment = str(asp.get("sentiment", "")).strip().lower()
        if sentiment not in SENTIMENT_LABELS:
            sentiment = "neutral"
        if category is None:
            continue
        seen.add(term)
        aspects.append({"term": term, "category": category, "sentiment": sentiment})
    overall = str(item.get("overall", "")).strip().lower()
    if overall not in SENTIMENT_LABELS:
        overall = ""
    return {"aspects": aspects, "overall": overall}


def weak_label(text: str, categories: Sequence[str]) -> Dict:
    """无 API 时的规则弱标注：方面词典匹配 + 情感词计数。"""
    from .preprocessor import NEGATIVE_WORDS, POSITIVE_WORDS
    from .taxonomy import lexicon_extract
    aspects = []
    for asp in lexicon_extract(text):
        term = asp["aspect_term"]
        window = text[max(0, text.find(term) - 6): text.find(term) + len(term) + 6]
        pos = sum(1 for w in POSITIVE_WORDS if w in window)
        neg = sum(1 for w in NEGATIVE_WORDS if w in window)
        sentiment = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
        aspects.append({"term": term, "category": asp["aspect_category"], "sentiment": sentiment})
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    overall = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
    return {"aspects": aspects, "overall": overall}


def annotate_batch(texts: Sequence[str], categories: Sequence[str],
                   client: Optional[LLMClient]) -> List[Dict]:
    """标注一批文本；返回与 texts 等长的标注列表。"""
    if client is None:
        return [weak_label(t, categories) for t in texts]
    content = client.chat(
        [{"role": "system", "content": ANNOTATION_SYSTEM},
         {"role": "user", "content": build_annotation_prompt(texts, categories)}],
        temperature=0.0, max_tokens=200 + 160 * len(texts), json_mode=True,
    )
    items = parse_json_array(content)
    if not items:
        return [{"aspects": [], "overall": "", "annotator": "failed"} for _ in texts]
    by_id = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = item.get("id", idx + 1)
        try:
            key = int(key)
        except Exception:
            key = idx + 1
        by_id[key] = item
    out = []
    for i, text in enumerate(texts):
        item = by_id.get(i + 1, {})
        cleaned = _clean_annotation(item, text, categories)
        cleaned["annotator"] = "llm" if item else "failed"
        out.append(cleaned)
    return out


def annotate_records(records: Sequence[Dict], client: Optional[LLMClient] = None,
                     categories: Optional[Sequence[str]] = None, batch_size: int = 6,
                     workers: int = 3, cache_path: Optional[str] = None,
                     verbose: bool = True) -> List[Dict]:
    """
    批量标注并写缓存（断点续标）。

    records: [{"id": int, "text": str, ...}]
    返回：records（就地补充 aspects / overall_sentiment / annotator 字段）

    缓存键使用「文本内容哈希」而不是 id：这样即使下次运行改变了抽样规模、
    评论的 id 发生变化，旧标注依然能正确复用，不会串行到别的评论上。
    """
    categories = list(categories or ASPECT_CATEGORY_LIST)

    def cache_key(text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:16]

    cache: Dict[str, Dict] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("key"):        # 旧版按 id 存储的缓存不再复用（避免张冠李戴）
                        cache[rec["key"]] = rec
    todo = [r for r in records if cache_key(r["text"]) not in cache]
    if verbose:
        print(f"  标注缓存命中 {len(records) - len(todo)} 条，本次需标注 {len(todo)} 条"
              f" | 标注器：{'LLM' if client else '规则弱标注'}")
    chunks = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    lock = threading.Lock()
    writer = open(cache_path, "a", encoding="utf-8") if cache_path else None

    def work(chunk: List[Dict]) -> List[Dict]:
        anns = annotate_batch([c["text"] for c in chunk], categories, client)
        out = []
        for rec, ann in zip(chunk, anns):
            item = {"key": cache_key(rec["text"]), "id": rec["id"], **ann}
            out.append(item)
        with lock:
            if writer:
                for item in out:
                    writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.flush()
        time.sleep(0.1)  # 轻微限速，避免触发服务端 QPS 限制
        return out

    done = 0
    if chunks:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, chunk) for chunk in chunks]
            for fut in as_completed(futures):
                for item in fut.result():
                    cache[item["key"]] = item
                done += 1
                if verbose and (done % 20 == 0 or done == len(chunks)):
                    print(f"  标注进度：{done}/{len(chunks)} 批")
    if writer:
        writer.close()

    failed = 0
    for rec in records:
        ann = cache.get(cache_key(rec["text"]))
        if ann is None:
            ann = {"aspects": [], "overall": "", "annotator": "missing"}
        rec["aspects"] = ann.get("aspects", [])
        rec["overall_sentiment"] = ann.get("overall") or rec.get("star_label") or "neutral"
        rec["annotator"] = ann.get("annotator", "missing")
        if ann.get("annotator") == "failed":
            failed += 1
    if verbose:
        n_aspect = sum(len(r["aspects"]) for r in records)
        print(f"  标注完成：{len(records)} 条评论，共 {n_aspect} 个方面实例，"
              f"平均 {n_aspect / max(len(records), 1):.2f} 个/条，失败 {failed} 条")
    return list(records)
