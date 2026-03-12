from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from markupsafe import Markup
from sentence_transformers import SentenceTransformer

# ===== Multilingual dictionary =====
I18N = {
    "zh": {
        "site_name" : "CPRet：编程竞赛题目检索",
        "new_domain_info": "我们的最新域名是 <a href='https://cpret.online' target='_blank' class='alert-link'>cpret.online</a>，我们的 GitHub 仓库是 <a href='https://github.com/coldchair/CPRet' target='_blank' class='alert-link'>CPRet</a>，欢迎收藏或 star！",
        "paper_info": "📰 2025 年 9 月更新：🎉 恭喜！我们的项目论文 <a href='https://neurips.cc/virtual/2025/poster/121814' target='_blank'>CPRet</a> 被 NeurIPS 2025 D&B track 接收！",
        "info": "📢 2025 年 7 月更新：我们已升级模型并同步更新了题目数据库，检索效果更佳！",
        "info2": "📢 2026 年 02 月更新：更新了题目库。",
        "placeholder": "输入题目描述或简略题意（超过 2048 token 的查询将被截断）…",
        "template_btn": "填入示例查询",
        "search_btn": "搜索",
        "summary"   : "共 <strong>{total}</strong> 条结果，页 {page}/{max_page}，耗时 {elapsed:.1f} ms",
        "prev"      : "上一页",
        "next"      : "下一页",
        "untitled"  : "未命名",
        "view_origin": "原站链接",
        "back": "返回搜索",
        "view_stats": "📊 每日搜索统计",
        "date": "日期",
        "search_count": "搜索次数",
        "total_search_count": "总搜索次数",
        "example_report": "使用示例（实测报告）",
        "filter_by_oj": "筛选 OJ",
        "select_all": "全选",      
        "deselect_all": "全不选",
        "moving_average": "渐近平均",
        "search_progress": "搜索进度",
        "elapsed_time": "已花费时间",
        "estimated_time": "预估时间",
        "token_count_label": "Token 数量",
    },
    "en": {
        "site_name" : "CPRet: Competitive Programming Problem Retrieval",
        "new_domain_info": "Our new domain is <a href='https://cpret.online' target='_blank' class='alert-link'>cpret.online</a>. Our GitHub repo is <a href='https://github.com/coldchair/CPRet' target='_blank' class='alert-link'>CPRet</a>. Please bookmark or star it!",
        "paper_info": "📰 September 2025 Update: 🎉 Congrats! Our project paper <a href='https://neurips.cc/virtual/2025/poster/121814' target='_blank'>CPRet</a> has been accepted by the NeurIPS 2025 D&B track!",
        "info": "📢 July 2025 Update: We've upgraded our model and synchronized the problem database for better retrieval!",
        "info2": "📢 Feb 2026 Update: Updated the problem database.",
        "placeholder": "Enter problem description or simplified statement (queries longer than 2048 tokens will be truncated)…",
        "template_btn": "Insert example query",
        "search_btn": "Search",
        "summary"   : "<strong>{total}</strong> results, page {page}/{max_page}, {elapsed:.1f} ms",
        "prev"      : "Prev",
        "next"      : "Next",
        "untitled"  : "Untitled",
        "view_origin": "Original Link",
        "back": "Back to Search",
        "view_stats": "📊 Daily Search Stats",
        "date": "Date",
        "search_count": "Search Count",
        "total_search_count": "Total Search Count",
        "example_report": "Test Cases (Demo Report)",
        "filter_by_oj": "Filter by OJ",
        "select_all": "Select All",      
        "deselect_all": "Deselect All",
        "moving_average": "Moving Average",
        "search_progress": "Search Progress",
        "elapsed_time": "Elapsed",
        "estimated_time": "Estimated",
        "token_count_label": "Token Count",
    },
}


def detect_lang():
    """Language priority: ?lang= -> Accept-Language -> zh"""
    qlang = request.args.get("lang")
    if qlang in ("zh", "en"):
        return qlang
    header = request.headers.get("Accept-Language", "")
    return "en" if header.lower().startswith("en") else "zh"


# ---------------- Configuration ---------------- #
SEARCH_STATS_PATH = "search_stats.json"
SEARCH_TIME_STATS_PATH = "search_time_stats.json"
MODEL_PATH = os.getenv("MODEL_PATH", "coldchair16/CPRetriever-Prob-Qwen3-4B-2510")
EMB_PATH = os.getenv("EMB_PATH", "./probs_2602_embs.npy")
PROB_PATH = os.getenv("PROB_PATH", "./probs_2602.jsonl")

PAGE_SIZE = 20
MAX_QUERY_TOKENS = 2048
SEARCH_CACHE_SIZE = 1024
MAX_TIMING_SAMPLES = 5000
# ------------------------------------- #


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


import tempfile
def save_json(path: str, payload: Any) -> None:
    dirname = os.path.dirname(os.path.abspath(path))
    fd, temp_path = tempfile.mkstemp(dir=dirname, text=True)
    try:
        with os.fdopen(fd, 'w', encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path) 
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def parse_page(raw_page: str) -> int:
    try:
        return max(int(raw_page), 1)
    except (TypeError, ValueError):
        return 1

app = Flask(__name__)

# ---------- Load model & data on startup ---------- #
print("Loading SentenceTransformer model …")
use_bf16 = env_flag("BF_16", default=True)
if use_bf16:
    model = SentenceTransformer(
        MODEL_PATH,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
else:
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.tokenizer.model_max_length = MAX_QUERY_TOKENS
model.max_seq_length = MAX_QUERY_TOKENS

print("Loading pre‑computed embeddings …")
embs = np.load(EMB_PATH).astype("float32")
embs /= np.linalg.norm(embs, axis=1, keepdims=True)

print("Loading problem metadata …")
with open(PROB_PATH, "r", encoding="utf-8") as f:
    probs = [json.loads(line) for line in f]

# 收集所有可用的 OJ 源，用于前端下拉菜单
available_ojs = sorted({p.get("source") for p in probs if p.get("source")})

assert len(probs) == embs.shape[0], "Mismatch between vector and problem count!"
print(f"Ready! {len(probs)} problems indexed.\n")


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def search_once(q: str) -> Tuple[List[int], List[float]]:
    """Return ranked indices and similarity list (numpy array -> Python list)"""
    q_emb = model.encode(q, convert_to_tensor=True).to(torch.float32).cpu().numpy()
    q_emb = q_emb / np.linalg.norm(q_emb)
    sims = embs.dot(q_emb)
    idx = sims.argsort()[::-1]
    return idx.tolist(), sims.tolist()


def load_search_stats() -> Dict[str, int]:
    """Load daily search statistics from file."""
    stats = load_json(SEARCH_STATS_PATH, {})
    return stats if isinstance(stats, dict) else {}


def save_search_stats(stats: Dict[str, int]) -> None:
    """Save search statistics to file."""
    save_json(SEARCH_STATS_PATH, stats)


def record_search() -> None:
    """Update today's search count and save to file."""
    stats = load_search_stats()
    today = datetime.now().strftime("%Y-%m-%d")
    stats[today] = stats.get(today, 0) + 1
    save_search_stats(stats)


def load_search_time_stats() -> List[Dict[str, Any]]:
    """Load search timing samples from file."""
    data = load_json(SEARCH_TIME_STATS_PATH, [])
    return data if isinstance(data, list) else []


def save_search_time_stats(samples: Sequence[Dict[str, Any]]) -> None:
    """Save search timing samples to file."""
    save_json(SEARCH_TIME_STATS_PATH, list(samples))


def record_search_timing(token_count: int, elapsed_ms: float) -> None:
    """Append one timing sample used for ETA regression."""
    samples = load_search_time_stats()
    samples.append(
        {
            "token_count": int(token_count),
            "elapsed_ms": float(elapsed_ms),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    if len(samples) > MAX_TIMING_SAMPLES:
        samples = samples[-MAX_TIMING_SAMPLES:]
    save_search_time_stats(samples)


def fit_eta_model(samples: Sequence[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Fit ETA model: y = a*x + b
    x: query token count
    y: elapsed milliseconds
    The first valid sample is skipped due to warmup overhead.
    """
    valid: List[Tuple[int, float]] = []
    for s in samples:
        try:
            x = int(s.get("token_count", 0))
            y = float(s.get("elapsed_ms", 0.0))
        except (TypeError, ValueError):
            continue
        if x > 0 and y > 0:
            valid.append((x, y))

    if len(valid) <= 1:
        return None

    train = valid[1:]  # Skip first sample by requirement.
    xs = np.asarray([x for x, _ in train], dtype=np.float64)
    ys = np.asarray([y for _, y in train], dtype=np.float64)

    if len(train) == 1:
        a = 0.0
        b = float(ys[0])
    else:
        x_mean = xs.mean()
        y_mean = ys.mean()
        denom = np.sum((xs - x_mean) ** 2)
        if denom <= 1e-12:
            a = 0.0
            b = float(y_mean)
        else:
            a = float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)
            b = float(y_mean - a * x_mean)
            # Avoid obviously unstable negative values.
            if a < 0:
                a = 0.0
                b = float(y_mean)

    if b < 0:
        b = 0.0
    return {"a": a, "b": b, "train_size": len(train)}


def estimate_elapsed_ms(token_count: int, model_ab: Optional[Dict[str, float]]) -> Optional[float]:
    """Predict elapsed time in ms using y = a*x + b."""
    if not model_ab:
        return None
    estimated = model_ab["a"] * float(token_count) + model_ab["b"]
    return max(0.0, float(estimated))


def get_query_token_count(q: str) -> int:
    """Get tokenizer token count (same tokenizer used by retrieval model)."""
    encoded = model.tokenizer(
        q,
        add_special_tokens=False,
        truncation=True,
        max_length=model.tokenizer.model_max_length,
    )
    input_ids = encoded.get("input_ids", [])
    return len(input_ids)


def get_selected_ojs() -> List[str]:
    if "oj" in request.args:
        return request.args.getlist("oj")
    return available_ojs


def filter_indices_by_oj(indices: Sequence[int], selected_ojs: Sequence[str]) -> List[int]:
    selected = set(selected_ojs)
    return [j for j in indices if probs[j].get("source") in selected]


def build_results_page(
    filtered_indices: Sequence[int],
    sims: Sequence[float],
    page: int,
    title_fallback: str,
) -> List[Dict[str, Any]]:
    start = (page - 1) * PAGE_SIZE
    end = page * PAGE_SIZE
    page_indices = filtered_indices[start:end]

    results: List[Dict[str, Any]] = []
    for rank, j in enumerate(page_indices, start=start + 1):
        problem = probs[j]
        results.append(
            {
                "rank": rank,
                "pid": j,
                "score": float(sims[j]),
                "title": problem.get("title") or title_fallback,
                "url": problem.get("url", "#"),
                "source": problem.get("source", ""),
            }
        )
    return results


@app.route("/", methods=["GET"])
def index() -> str:
    lang = detect_lang()
    t = I18N[lang]

    q = request.args.get("q", "").strip()
    page = parse_page(request.args.get("page", "1"))
    selected_ojs = get_selected_ojs()

    results, total, elapsed = [], 0, 0.0
    query_token_count = 0
    estimated_ms = None
    time_samples = load_search_time_stats()
    eta_model = fit_eta_model(time_samples)

    if q:
        query_token_count = get_query_token_count(q)
        estimated_ms = estimate_elapsed_ms(query_token_count, eta_model)
        record_search()
        tic   = time.perf_counter()
        idx, sims = search_once(q)
        elapsed = (time.perf_counter() - tic) * 1_000
        record_search_timing(query_token_count, elapsed)

        filtered_idx = filter_indices_by_oj(idx, selected_ojs)
        total = len(filtered_idx)
        results = build_results_page(filtered_idx, sims, page, t["untitled"])

    return render_template(
        "index.html",
        lang=lang,
        t=t,
        query=q,
        results=results,
        page=page,
        page_size=PAGE_SIZE,
        total=total,
        max_page=max(1, math.ceil(total / PAGE_SIZE)),
        elapsed=elapsed,
        query_token_count=query_token_count,
        estimated_ms=estimated_ms,
        estimator_a=(eta_model["a"] if eta_model else 0.0),
        estimator_b=(eta_model["b"] if eta_model else 0.0),
        estimator_ready=(1 if eta_model else 0),
        available_ojs=available_ojs,
        selected_ojs=selected_ojs,
    )


@app.route("/estimate_eta", methods=["POST"])
def estimate_eta():
    """Return exact tokenizer token count and ETA for an input query."""
    payload = request.get_json(silent=True) or {}
    q = str(payload.get("q", "")).strip()
    if not q:
        return jsonify({"token_count": 0, "estimated_ms": None})

    token_count = get_query_token_count(q)
    eta_model = fit_eta_model(load_search_time_stats())
    estimated_ms = estimate_elapsed_ms(token_count, eta_model)
    return jsonify(
        {
            "token_count": token_count,
            "estimated_ms": estimated_ms,
            "a": (eta_model["a"] if eta_model else None),
            "b": (eta_model["b"] if eta_model else None),
            "estimator_ready": bool(eta_model),
        }
    )


@app.route("/p/<int:pid>")
def problem(pid: int):
    lang = detect_lang()
    t = I18N[lang]

    if pid < 0 or pid >= len(probs):
        return f"Problem #{pid} not found", 404

    p = probs[pid]

    raw = p.get("text", "(No text)").replace("\n", "<br>")
    text_html = Markup(raw)

    return render_template(
        "problem.html",
        lang=lang,
        t=t,
        pid=pid,
        title=p.get("title") or t["untitled"],
        source=p.get("source", ""),
        url=p.get("url", "#"),
        text_html=text_html,
        query=request.args.get("q", ""),
        page=request.args.get("page", "1"),
        selected_ojs_str=request.args.get("oj", ""),
    )


@app.route("/stats")
def stats():
    lang = detect_lang()
    t = I18N[lang]

    stats = load_search_stats()
    stats_data = sorted(stats.items(), key=lambda x: x[0], reverse=True)

    stats_draw = list(reversed(stats_data))
    stats_draw = stats_draw[:-1]

    return render_template(
        "stats.html",
        lang=lang,
        t=t,
        stats=stats_data,
        stats_draw=stats_draw,
    )

# -------------- Local run entry -------------- #
if __name__ == "__main__":
    # export FLASK_ENV=development to enable auto-reload/debug
    app.run(host="0.0.0.0", port=5000, debug=False)
