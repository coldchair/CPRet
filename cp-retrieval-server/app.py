from __future__ import annotations

import json
import math
import os
import secrets
import tempfile
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from flask import Flask, g, jsonify, render_template, request, session, url_for
from markupsafe import Markup, escape
from werkzeug.datastructures import MultiDict

from retrieval.example_queries import EXAMPLE_QUERIES
from retrieval.search_context import SearchContextStore

import retrieval.insights as insights_mod

# ===== Multilingual dictionary =====
I18N = {
    "zh": {
        "site_name" : "CPRet：编程竞赛题目检索",
        "site_short": "CPRet",
        "tagline": "用自然语言描述题意，在十余万道编程竞赛题中找到相似题",
        "new_domain_info": "最新域名 <a href='https://cpret.online' target='_blank' class='alert-link'>cpret.online</a> · <a href='https://github.com/coldchair/CPRet' target='_blank' class='alert-link'>GitHub</a> · 题目库 <a href='https://huggingface.co/datasets/coldchair16/CPRet-Embeddings' target='_blank' class='alert-link'>Hugging Face 下载</a>",
        "paper_info": "📰 2025 年 9 月更新：🎉 恭喜！我们的项目论文 <a href='https://neurips.cc/virtual/2025/poster/121814' target='_blank'>CPRet</a> 被 NeurIPS 2025 D&B track 接收！",
        "info": "📢 2025 年 7 月更新：我们已升级模型并同步更新了题目数据库，检索效果更佳！",
        "info2": "📢 2026 年 11 月更新：已更新题目库。",
        "placeholder": "输入题目描述或简略题意（超过 {max_tokens} token 的查询将被截断）…",
        "template_btn": "示例查询",
        "example_idea": "思路",
        "example_short": "简化题意",
        "example_full": "完整题意",
        "example_select": "选择示例题目",
        "level_easy": "简单",
        "level_medium": "中等",
        "level_hard": "困难",
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
        # --- new: insights & UI ---
        "insights_title": "结果方向",
        "insights_hint": "基于前 {k} 条结果的题目向量自动分析",
        "clusters_title": "方向分组",
        "clusters_empty": "前列结果方向较集中，未拆分分组。",
        "scatter_title": "二维分布",
        "scatter_hint": "越靠近 ★ 越接近查询；同色为同一方向，点击查看题面",
        "query_point": "查询",
        "problems_unit": "题",
        "avg_sim": "平均",
        "max_sim": "最高",
        "mmr_label": "多样化排序",
        "mmr_tip": "让靠前结果覆盖不同方向，减少近似题扎堆",
        "dedup_label": "合并重复题",
        "dedup_tip": "将不同 OJ 上的同一道题折叠为一条",
        "see_also": "另有 {n} 个同题/近似题",
        "cluster_filter_active": "仅显示方向",
        "clear_filter": "清除筛选",
        "show_all": "全部",
        "view_problem": "查看题面",
        "orig_rank": "原排名",
        "selected_count": "已选 {n}/{total}",
        "recent7": "近 7 日日均",
        "recent30": "近 30 日日均",
        "days_tracked": "统计天数",
        "stats_table": "每日明细",
        "chart_note": "图表不含当天（未结束）",
        "no_data": "暂无数据",
        "mock_banner": "调试模式：未加载模型，查询向量由已有题目 embedding 近似生成",
        "dup_groups": "合并 {n} 组重复",
    },
    "en": {
        "site_name" : "CPRet: Competitive Programming Problem Retrieval",
        "site_short": "CPRet",
        "tagline": "Describe a problem in natural language and find similar ones among 100k+ competitive programming problems",
        "new_domain_info": "Domain <a href='https://cpret.online' target='_blank' class='alert-link'>cpret.online</a> · <a href='https://github.com/coldchair/CPRet' target='_blank' class='alert-link'>GitHub</a> · dataset <a href='https://huggingface.co/datasets/coldchair16/CPRet-Embeddings' target='_blank' class='alert-link'>Hugging Face</a>",
        "paper_info": "📰 September 2025 Update: 🎉 Congrats! Our project paper <a href='https://neurips.cc/virtual/2025/poster/121814' target='_blank'>CPRet</a> has been accepted by the NeurIPS 2025 D&B track!",
        "info": "📢 July 2025 Update: We've upgraded our model and synchronized the problem database for better retrieval!",
        "info2": "📢 November 2026 Update: Problem database refreshed.",
        "placeholder": "Enter problem description or simplified statement (queries longer than {max_tokens} tokens will be truncated)…",
        "template_btn": "Examples",
        "example_idea": "Idea",
        "example_short": "Short statement",
        "example_full": "Full statement",
        "example_select": "Choose an example problem",
        "level_easy": "Easy",
        "level_medium": "Medium",
        "level_hard": "Hard",
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
        # --- new: insights & UI ---
        "insights_title": "Result Directions",
        "insights_hint": "Automatically analysed from embeddings of the top {k} results",
        "clusters_title": "Topic Groups",
        "clusters_empty": "Top results are focused on one direction; no grouping.",
        "scatter_title": "2D Map",
        "scatter_hint": "Closer to ★ means closer to the query; same colour = same group; click to open",
        "query_point": "Query",
        "problems_unit": "problems",
        "avg_sim": "avg",
        "max_sim": "max",
        "mmr_label": "Diversify",
        "mmr_tip": "Make top results cover different directions instead of near-duplicates",
        "dedup_label": "Merge duplicates",
        "dedup_tip": "Fold the same problem on different OJs into one entry",
        "see_also": "{n} more identical/similar",
        "cluster_filter_active": "Showing group",
        "clear_filter": "Clear filter",
        "show_all": "All",
        "view_problem": "Statement",
        "orig_rank": "Orig. rank",
        "selected_count": "{n}/{total} selected",
        "recent7": "7-day avg",
        "recent30": "30-day avg",
        "days_tracked": "Days tracked",
        "stats_table": "Daily Details",
        "chart_note": "Chart excludes today (incomplete)",
        "no_data": "No data yet",
        "mock_banner": "Debug mode: model not loaded; query vectors are approximated from stored problem embeddings",
        "dup_groups": "{n} duplicate groups merged",
    },
}

# Keys exposed to front-end JS.
JS_I18N_KEYS = (
    "token_count_label", "problems_unit", "avg_sim", "max_sim", "query_point",
    "clear_filter", "cluster_filter_active", "clusters_empty", "insights_hint",
    "search_count", "moving_average", "date", "no_data", "selected_count",
    "view_problem", "show_all",
)


def detect_lang():
    """Language priority: ?lang= -> Accept-Language -> zh"""
    qlang = g.search_args.get("lang") if "search_args" in g else request.args.get("lang")
    if qlang in ("zh", "en"):
        return qlang
    header = request.headers.get("Accept-Language", "")
    return "en" if header.lower().startswith("en") else "zh"


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_number(name: str, default: float, lo: float, hi: float, cast=float):
    raw = os.getenv(name)
    try:
        val = cast(raw) if raw is not None else default
    except (TypeError, ValueError):
        val = default
    return cast(min(max(val, lo), hi))


# ---------------- Configuration ---------------- #
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_MODEL = env_flag("MOCK_MODEL", default=False)
_STATS_SUFFIX = ".mock" if MOCK_MODEL else ""
SEARCH_STATS_PATH = os.path.join(_APP_DIR, "search_stats.json" + _STATS_SUFFIX)
MODEL_PATH = os.getenv("MODEL_PATH", "coldchair16/CPRetriever-Prob-Qwen3-4B-2510")
ENCODER_BACKEND = os.getenv("ENCODER_BACKEND", "sentence_transformers").strip().lower()
_TIMING_SUFFIX = _STATS_SUFFIX or (".vllm" if ENCODER_BACKEND == "vllm" else "")
SEARCH_TIME_STATS_PATH = os.path.join(_APP_DIR, "search_time_stats.json" + _TIMING_SUFFIX)
EMB_PATH = os.getenv("EMB_PATH", "./probs_2611_embs.npy")
PROB_PATH = os.getenv("PROB_PATH", "./probs_2611.jsonl")
# Memory-map embeddings instead of loading a normalised copy (saves RAM).
EMB_MMAP = env_flag("EMB_MMAP", default=MOCK_MODEL)

PAGE_SIZE = 20
MAX_QUERY_TOKENS = int(env_number("MAX_QUERY_TOKENS", 2048, 2048, 8192, int))
ENGINE_MAX_QUERY_TOKENS = int(env_number("ENGINE_MAX_QUERY_TOKENS", 8192, MAX_QUERY_TOKENS, 8192, int))
LOCAL_ADMIN_ENABLED = env_flag("LOCAL_ADMIN", default=False)
LOCAL_ADMIN_SOCKET = os.getenv("LOCAL_ADMIN_SOCKET", os.path.join(_APP_DIR, ".runtime", "admin.sock"))
DATA_UPDATE_ROOT = os.getenv("DATA_UPDATE_ROOT", os.path.join(_APP_DIR, "updates"))
SEARCH_CACHE_SIZE = int(env_number("SEARCH_CACHE_SIZE", 1024, 1, 100000, int))
INSIGHT_CACHE_SIZE = 256
MAX_TIMING_SAMPLES = 5000
# Ignore sub-millisecond timings (LRU cache hits) when fitting ETA.
MIN_TIMING_RECORD_MS = 50.0

INSIGHT_TOPK = int(env_number("INSIGHT_TOPK", 100, 10, 200, int))
DEDUP_THRESHOLD = float(env_number("DEDUP_THRESHOLD", 0.92, 0.5, 1.0))
MMR_LAMBDA = float(env_number("MMR_LAMBDA", 0.7, 0.0, 1.0))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(env_number("PORT", 5000, 1, 65535, int))
# ------------------------------------- #


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


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


def parse_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_cluster(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v < 64 else None


_search_stats_lock = threading.Lock()
_search_stats_mem: Dict[str, int] = {}


def _coerce_stats_dict(data: Any) -> Dict[str, int]:
    if not isinstance(data, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _read_search_stats_file() -> Dict[str, int]:
    raw = load_json(SEARCH_STATS_PATH, {})
    return _coerce_stats_dict(raw)


def _write_search_stats_file(payload: Dict[str, int]) -> None:
    save_json(SEARCH_STATS_PATH, payload)


def init_search_stats_memory() -> None:
    global _search_stats_mem
    loaded = _read_search_stats_file()
    with _search_stats_lock:
        _search_stats_mem = loaded


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
_search_contexts = SearchContextStore()


from encoder.async_pooler import EncoderBusyError, EncoderTimeoutError


@app.errorhandler(EncoderBusyError)
def encoder_busy(error):
    return jsonify({"error": "encoder_busy"}), 503, {"Retry-After": "1"}


@app.errorhandler(EncoderTimeoutError)
def encoder_timeout(error):
    return jsonify({"error": "encoder_timeout"}), 504


def query_owner() -> str:
    if "query_owner" not in session:
        session["query_owner"] = secrets.token_urlsafe(24)
    return session["query_owner"]


def search_request_args() -> MultiDict:
    if request.method != "POST":
        return MultiDict(request.args)
    if not request.is_json:
        return MultiDict(request.form)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("invalid_search_payload")
    args = MultiDict()
    for key in ("q", "qid", "lang", "page", "mmr", "dedup", "cluster", "count", "oj"):
        if key not in payload or payload[key] is None:
            continue
        values = payload[key] if key == "oj" and isinstance(payload[key], list) else [payload[key]]
        if key == "oj" and not values:
            args.add("oj", "")
        if len(values) > 100:
            raise ValueError("invalid_search_payload")
        for value in values:
            if not isinstance(value, (str, int, bool)) or (key in ("q", "qid", "lang", "oj") and not isinstance(value, str)):
                raise ValueError("invalid_search_payload")
            args.add(key, "1" if value is True else "0" if value is False else str(value))
    return args


def resolve_search_query(args) -> Tuple[str, Optional[str]]:
    qid = args.get("qid")
    if "q" in args:
        query = args.get("q", "").strip()
    elif qid:
        query = _search_contexts.get(query_owner(), qid)
        if query is None:
            raise ValueError("search_context_expired")
    else:
        query = ""
    if query:
        qid = _search_contexts.put(query_owner(), query, qid)
    else:
        qid = None
    return query, qid

# ---------- Load model & data on startup ---------- #
from retrieval.retrieval_state import ReadWriteLock, load_problem_data, model_fingerprint

model = None
mock_encoder = None
emb_norms: Optional[np.ndarray] = None
_retrieval_lock = ReadWriteLock()
_hot_controller = None
_local_admin = None


def activate_problem_data(data, query_tokens: int) -> None:
    global embs, emb_norms, probs, available_ojs, _oj_code, source_codes, MAX_QUERY_TOKENS
    global SEARCH_TIME_STATS_PATH
    embs, emb_norms, probs = data.embeddings, data.norms, data.probs
    available_ojs, _oj_code, source_codes = data.available_ojs, data.oj_code, data.source_codes
    MAX_QUERY_TOKENS = query_tokens
    timing_suffix = _TIMING_SUFFIX + (".8192" if query_tokens == 8192 else "")
    SEARCH_TIME_STATS_PATH = os.path.join(_APP_DIR, "search_time_stats.json" + timing_suffix)
    encode_query.cache_clear()
    rank_all.cache_clear()
    get_insights.cache_clear()


def initialize_retriever() -> None:
    global model, mock_encoder, _hot_controller, _retrieval_lock, _local_admin

    if MOCK_MODEL and LOCAL_ADMIN_ENABLED:
        raise ValueError("Local hot updates require a real encoder, not MOCK_MODEL")
    print("Loading embeddings and problem metadata …")
    data = load_problem_data(PROB_PATH, EMB_PATH, mmap=EMB_MMAP, allow_empty_text=True)

    if MOCK_MODEL:
        print("[MOCK_MODEL] Skipping model loading (debug only).")
    else:
        from encoder.query_encoder import create_encoder

        print(f"Loading {ENCODER_BACKEND} model …")
        model = create_encoder(
            MODEL_PATH,
            backend=ENCODER_BACKEND,
            max_seq_length=MAX_QUERY_TOKENS,
            use_bf16=env_flag("BF_16", default=True),
            engine_max_seq_length=ENGINE_MAX_QUERY_TOKENS,
        )
        if hasattr(model, "close"):
            import atexit

            atexit.register(model.close)

    activate_problem_data(data, MAX_QUERY_TOKENS)
    if MOCK_MODEL:
        from encoder.mock_encoder import MockEncoder

        mock_encoder = MockEncoder(probs, get_vecs)
    if LOCAL_ADMIN_ENABLED:
        import atexit
        from pathlib import Path
        from admin.hot_update import HotUpdateController
        from admin.local_admin import LocalAdminServer

        _hot_controller = HotUpdateController(
            data, model, activate_problem_data, model_fingerprint(MODEL_PATH),
            DATA_UPDATE_ROOT, Path(LOCAL_ADMIN_SOCKET).parent / "active.json", MAX_QUERY_TOKENS,
        )
        _retrieval_lock = _hot_controller.lock
        _hot_controller.restore()
        _local_admin = LocalAdminServer(LOCAL_ADMIN_SOCKET, _hot_controller.handle)
        atexit.register(_local_admin.close)
    print(f"Ready! {len(probs)} problems indexed.\n")
    init_search_stats_memory()


@app.before_request
def acquire_retrieval_snapshot():
    if request.endpoint == "estimate_eta":
        payload = request.get_json(silent=True) or {}
        g.eta_payload = payload if isinstance(payload, dict) else {}
    if request.endpoint in ("api_search", "index"):
        try:
            g.search_args = search_request_args()
            g.query, g.query_id = resolve_search_query(g.search_args)
        except ValueError as error:
            if request.endpoint == "api_search":
                return jsonify({"error": str(error)}), 410 if str(error) == "search_context_expired" else 400
            g.search_args = MultiDict()
            g.query, g.query_id = "", None
            g.query_error = str(error)
    _retrieval_lock.acquire_read()
    g.retrieval_read_locked = True


@app.teardown_request
def release_retrieval_snapshot(error=None) -> None:
    if g.pop("retrieval_read_locked", False):
        _retrieval_lock.release_read()


def get_vecs(idx: np.ndarray) -> np.ndarray:
    """Return L2-normalised float32 embeddings for the given indices."""
    idx = np.asarray(idx, dtype=np.int64)
    v = np.asarray(embs[idx], dtype=np.float32)
    if emb_norms is not None:
        v = v / emb_norms[idx, None]
    return v


def score_all(q_emb: np.ndarray) -> np.ndarray:
    sims = np.asarray(embs.dot(q_emb), dtype=np.float32)
    if emb_norms is not None:
        sims = sims / emb_norms
    return sims


@lru_cache(maxsize=SEARCH_CACHE_SIZE * 4)
def encode_query(q: str) -> np.ndarray:
    """Encode and L2-normalise the query (float32, read-only)."""
    if mock_encoder is not None:
        q_emb = mock_encoder.encode(q)
    else:
        q_emb = model.encode(q)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb)
    q_emb.setflags(write=False)
    return q_emb


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def rank_all(q: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ranked indices int32, similarity float32) as read-only arrays."""
    q_emb = encode_query(q)
    sims = score_all(q_emb)
    order = np.argsort(-sims, kind="stable").astype(np.int32)
    order.setflags(write=False)
    sims.setflags(write=False)
    return order, sims


def load_search_stats() -> Dict[str, int]:
    """Daily search statistics from in-memory cache (copy)."""
    with _search_stats_lock:
        return dict(_search_stats_mem)


def record_search() -> None:
    """Increment today's count in memory and persist (threading.Lock + save_json)."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _search_stats_lock:
        _search_stats_mem[today] = _search_stats_mem.get(today, 0) + 1
        _write_search_stats_file(dict(_search_stats_mem))


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
        except (TypeError, ValueError, AttributeError):
            continue
        if x > 0 and y >= MIN_TIMING_RECORD_MS:
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
    if mock_encoder is not None:
        return mock_encoder.count_tokens(q, MAX_QUERY_TOKENS)
    return model.count_tokens(q)


def get_selected_ojs_from_args(args: Any) -> List[str]:
    if "oj" in args:
        allowed = set(available_ojs)
        seen = set()
        out = []
        for name in args.getlist("oj"):
            if name in allowed and name not in seen:
                seen.add(name)
                out.append(name)
        return out
    return available_ojs


def parse_view_params(args: Any) -> Dict[str, Any]:
    return {
        "mmr": parse_bool(args.get("mmr"), False),
        "dedup": parse_bool(args.get("dedup"), True),
        "cluster": parse_cluster(args.get("cluster")),
    }


def filter_indices_by_oj(indices: np.ndarray, oj_key: Tuple[str, ...]) -> np.ndarray:
    """Vectorised OJ filter preserving order."""
    if len(oj_key) == len(available_ojs):
        return indices
    codes = np.array([_oj_code[o] for o in oj_key], dtype=np.int32)
    mask = np.isin(source_codes, codes)
    return indices[mask[indices]]


def _title_of(pid: int, fallback: str = "") -> str:
    return probs[pid].get("title") or fallback


@lru_cache(maxsize=INSIGHT_CACHE_SIZE)
def get_insights(q: str, oj_key: Tuple[str, ...], dedup: bool) -> Dict[str, Any]:
    """Cluster / dedup / MMR / 2D analysis on the filtered top-K (cached)."""
    order, sims = rank_all(q)
    filtered = filter_indices_by_oj(order, oj_key)
    top = np.asarray(filtered[:INSIGHT_TOPK], dtype=np.int64)
    if top.size == 0:
        res = insights_mod.analyze([], np.zeros((0, 1)), np.zeros(0), [], None)
    else:
        tic = time.perf_counter()
        res = insights_mod.analyze(
            top,
            get_vecs(top),
            sims[top],
            [_title_of(int(i)) for i in top],
            encode_query(q),
            dedup_threshold=DEDUP_THRESHOLD,
            mmr_lambda=MMR_LAMBDA,
            dedup=dedup,
        )
        app.logger.debug("insights computed in %.1f ms", (time.perf_counter() - tic) * 1000)
    res["orig_rank"] = {int(pid): i + 1 for i, pid in enumerate(top)}
    return res


def public_insights(ins: Dict[str, Any], dup_groups: int) -> Dict[str, Any]:
    """Subset of insights sent to the browser."""
    return {
        "k": ins["k"],
        "clusters": ins["clusters"],
        "points": ins["points"] if ins["clusters"] or len(ins["points"]) >= 3 else [],
        "query_point": ins["query_point"],
        "dup_groups": dup_groups,
    }


def build_display_order(
    filtered: np.ndarray,
    ins: Dict[str, Any],
    mmr: bool,
    cluster: Optional[int],
) -> Tuple[np.ndarray, Optional[int]]:
    """Return the list to paginate and the effective cluster filter."""
    k = ins["k"]
    if k == 0:
        return filtered, None
    head = ins["mmr_order"] if mmr else ins["primary"]
    if cluster is not None and 0 <= cluster < len(ins["clusters"]):
        assign = ins["assign"]
        sel = [p for p in head if assign.get(p) == cluster]
        return np.asarray(sel, dtype=np.int64), cluster
    tail = np.asarray(filtered[k:], dtype=np.int64)
    return np.concatenate([np.asarray(head, dtype=np.int64), tail]), None


def execute_search(
    q: str,
    page: int,
    selected_ojs: List[str],
    title_fallback: str,
    view: Optional[Dict[str, Any]] = None,
    count: bool = True,
) -> Dict[str, Any]:
    """Run retrieval and return result payload for templates / API."""
    view = view or {"mmr": False, "dedup": True, "cluster": None}
    results: List[Dict[str, Any]] = []
    total = 0
    elapsed = 0.0
    query_token_count = 0
    estimated_ms: Optional[float] = None
    empty_insights = {"k": 0, "clusters": [], "points": [], "query_point": None, "dup_groups": 0}

    if not q:
        return {
            "results": results,
            "total": total,
            "elapsed": elapsed,
            "query_token_count": query_token_count,
            "estimated_ms": estimated_ms,
            "max_page": 1,
            "insights": empty_insights,
            "params": dict(view),
        }

    time_samples = load_search_time_stats()
    eta_model = fit_eta_model(time_samples)
    query_token_count = get_query_token_count(q)
    estimated_ms = estimate_elapsed_ms(query_token_count, eta_model)

    if count:
        record_search()
    cache_hits_before = rank_all.cache_info().hits
    tic = time.perf_counter()
    order, sims = rank_all(q)
    elapsed = (time.perf_counter() - tic) * 1_000
    cache_hit = rank_all.cache_info().hits > cache_hits_before
    if not cache_hit and elapsed >= MIN_TIMING_RECORD_MS and not MOCK_MODEL:
        record_search_timing(query_token_count, elapsed)

    oj_key = tuple(sorted(selected_ojs))
    filtered = filter_indices_by_oj(order, oj_key)
    ins = get_insights(q, oj_key, bool(view["dedup"]))
    display, eff_cluster = build_display_order(filtered, ins, bool(view["mmr"]), view["cluster"])

    total = int(display.shape[0])
    max_page = max(1, math.ceil(total / PAGE_SIZE))
    results = build_results_page(display, sims, page, title_fallback, ins, filtered)

    dup_groups = len(ins["duplicates"]) if view["dedup"] else 0
    params = {"mmr": bool(view["mmr"]), "dedup": bool(view["dedup"]), "cluster": eff_cluster}
    return {
        "results": results,
        "total": total,
        "elapsed": elapsed,
        "query_token_count": query_token_count,
        "estimated_ms": estimated_ms,
        "max_page": max_page,
        "insights": public_insights(ins, dup_groups),
        "params": params,
    }


def build_results_page(
    display: np.ndarray,
    sims: np.ndarray,
    page: int,
    title_fallback: str,
    ins: Dict[str, Any],
    filtered: np.ndarray,
) -> List[Dict[str, Any]]:
    start = (page - 1) * PAGE_SIZE
    end = page * PAGE_SIZE
    page_indices = display[start:end]
    k = ins["k"]
    orig_rank = ins.get("orig_rank", {})
    assign = ins.get("assign", {})
    dup_map = ins.get("duplicates", {})

    results: List[Dict[str, Any]] = []
    for rank, j in enumerate(page_indices, start=start + 1):
        j = int(j)
        problem = probs[j]
        o_rank = orig_rank.get(j)
        if o_rank is None:
            # Tail items keep their position in the filtered list.
            o_rank = k + (rank - (display.shape[0] - (filtered.shape[0] - k))) if k else rank
        dups = [
            {
                "pid": d,
                "title": probs[d].get("title") or title_fallback,
                "url": probs[d].get("url", "#"),
                "source": probs[d].get("source", ""),
                "score": float(sims[d]),
            }
            for d in dup_map.get(j, [])
        ]
        results.append(
            {
                "rank": rank,
                "orig_rank": int(o_rank),
                "pid": j,
                "score": float(sims[j]),
                "title": problem.get("title") or title_fallback,
                "url": problem.get("url", "#"),
                "source": problem.get("source", ""),
                "cluster": assign.get(j),
                "duplicates": dups,
            }
        )
    return results


def _common_context(lang: str) -> Dict[str, Any]:
    t = dict(I18N[lang])
    t["placeholder"] = t["placeholder"].format(max_tokens=MAX_QUERY_TOKENS)
    return {
        "lang": lang,
        "t": t,
        "js_i18n": {k: t[k] for k in JS_I18N_KEYS},
        "example_queries": EXAMPLE_QUERIES[lang],
        "mock_mode": MOCK_MODEL,
    }


@app.context_processor
def inject_static_helpers() -> Dict[str, Any]:
    def static_v(filename: str) -> str:
        path = os.path.join(app.static_folder or "", filename)
        try:
            v = int(os.path.getmtime(path))
        except OSError:
            v = 0
        return url_for("static", filename=filename, v=v)

    def lang_url(target: str) -> str:
        """Current page URL with ?lang= replaced (keeps other query args)."""
        args = (g.search_args if "search_args" in g else request.args).to_dict(flat=False)
        args.pop("q", None)
        args.pop("count", None)
        if g.get("query_id"):
            args["qid"] = [g.query_id]
        args["lang"] = [target]
        endpoint = request.endpoint or "index"
        try:
            return url_for(endpoint, **(request.view_args or {}), **args)
        except Exception:  # pragma: no cover - defensive
            return url_for("index", lang=target)

    return {"static_v": static_v, "lang_url": lang_url}


@app.template_filter("safe_url")
def safe_url(value: Any) -> str:
    """Allow only http(s) / relative URLs in href attributes."""
    s = str(value or "").strip()
    low = s.lower()
    if low.startswith(("http://", "https://", "/")) and not low.startswith("//"):
        return s
    return "#"


def _template_args(q: str, page: int, selected_ojs: List[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    all_selected = len(selected_ojs) == len(available_ojs)
    return {
        "query": q,
        "query_id": g.get("query_id"),
        "query_error": g.get("query_error"),
        "page": page,
        "page_size": PAGE_SIZE,
        "selected_ojs": selected_ojs,
        "oj_link": None if all_selected else selected_ojs,
        "available_ojs": available_ojs,
        **payload,
    }


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    lang = detect_lang()
    t = I18N[lang]

    args = g.search_args
    q = g.query
    page = parse_page(args.get("page", "1"))
    selected_ojs = get_selected_ojs_from_args(args)
    view = parse_view_params(args)
    count = parse_bool(args.get("count"), "q" in args)

    payload = execute_search(q, page, selected_ojs, t["untitled"], view, count)

    return render_template(
        "index.html",
        **_common_context(lang),
        **_template_args(q, page, selected_ojs, payload),
    )


@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    """JSON API accepting body-based long queries and legacy GET requests."""
    lang = detect_lang()
    t = I18N[lang]

    args = g.search_args
    q = g.query
    if not q:
        return jsonify({"error": "empty_query"}), 400

    page = parse_page(args.get("page", "1"))
    selected_ojs = get_selected_ojs_from_args(args)
    view = parse_view_params(args)
    count = parse_bool(args.get("count"), "q" in args)
    payload = execute_search(q, page, selected_ojs, t["untitled"], view, count)

    html = render_template(
        "_search_results.html",
        **_common_context(lang),
        **_template_args(q, page, selected_ojs, payload),
    )
    return jsonify(
        {
            "html": html,
            "query_id": g.query_id,
            "elapsed": payload["elapsed"],
            "query_token_count": payload["query_token_count"],
            "estimated_ms": payload["estimated_ms"],
            "total": payload["total"],
            "page": page,
            "max_page": payload["max_page"],
            "insights": payload["insights"],
            "params": payload["params"],
        }
    )


@app.route("/estimate_eta", methods=["POST"])
def estimate_eta():
    """Return exact tokenizer token count and ETA for an input query."""
    payload = g.eta_payload
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

    # Escape first, then turn newlines into <br> (prevents XSS from problem text).
    text_html = Markup(escape(p.get("text") or "(No text)").replace("\n", Markup("<br>")))

    oj_list = get_selected_ojs_from_args(request.args) if "oj" in request.args else None
    return render_template(
        "problem.html",
        **_common_context(lang),
        pid=pid,
        title=p.get("title") or t["untitled"],
        source=p.get("source", ""),
        url=p.get("url", "#"),
        text_html=text_html,
        query_id=request.args.get("qid"),
        page=parse_page(request.args.get("page", "1")),
        oj_link=oj_list,
        params=parse_view_params(request.args),
    )


@app.route("/stats")
def stats():
    lang = detect_lang()

    stats = load_search_stats()
    stats_data = sorted(stats.items(), key=lambda x: x[0], reverse=True)

    stats_draw = list(reversed(stats_data))
    stats_draw = stats_draw[:-1]

    total = sum(c for _, c in stats_data)
    today = datetime.now().date()

    def window_avg(days: int) -> float:
        start = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        vals = [c for d, c in stats_data if start <= d < end]
        return (sum(vals) / days) if vals else 0.0

    return render_template(
        "stats.html",
        **_common_context(lang),
        stats=stats_data,
        stats_draw=stats_draw,
        total=total,
        avg7=window_avg(7),
        avg30=window_avg(30),
        days=len(stats_data),
        query="",
    )


# -------------- Local run entry -------------- #
if __name__ != "__mp_main__":
    initialize_retriever()

if __name__ == "__main__":
    # export FLASK_ENV=development to enable auto-reload/debug
    app.run(host=HOST, port=PORT, debug=False)
