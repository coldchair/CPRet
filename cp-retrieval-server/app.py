import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import math
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from sentence_transformers import SentenceTransformer
import time
from flask import Flask, request, render_template, redirect, url_for
from markupsafe import Markup, escape
from datetime import datetime

# ===== Multilingual dictionary =====
I18N = {
    "zh": {
        "site_name" : "CPRet：编程竞赛题目检索",
        "new_domain_info": "我们的最新域名是 <a href='https://cpret.online' target='_blank' class='alert-link'>cpret.online</a>，我们的 GitHub 仓库是 <a href='https://github.com/coldchair/CPRet' target='_blank' class='alert-link'>CPRet</a>，欢迎收藏或 star！",
        "paper_info": "📰 2025 年 9 月更新：🎉 恭喜！我们的项目论文 <a href='https://neurips.cc/virtual/2025/poster/121814' target='_blank'>CPRet</a> 被 NeurIPS 2025 D&B track 接收！",
        "info": "📢 2025 年 7 月更新：我们已升级模型并同步更新了题目数据库，检索效果更佳！",
        "info2": "📢 2025 年 12 月更新：更新了题目库。",
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
    },
    "en": {
        "site_name" : "CPRet: Competitive Programming Problem Retrieval",
        "new_domain_info": "Our new domain is <a href='https://cpret.online' target='_blank' class='alert-link'>cpret.online</a>. Our GitHub repo is <a href='https://github.com/coldchair/CPRet' target='_blank' class='alert-link'>CPRet</a>. Please bookmark or star it!",
        "paper_info": "📰 September 2025 Update: 🎉 Congrats! Our project paper <a href='https://neurips.cc/virtual/2025/poster/121814' target='_blank'>CPRet</a> has been accepted by the NeurIPS 2025 D&B track!",
        "info": "📢 July 2025 Update: We've upgraded our model and synchronized the problem database for better retrieval!",
        "info2": "📢 December 2025 Update: Updated the problem database.",
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
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "coldchair16/CPRetriever-Prob-Qwen3-4B-2510"
)
EMB_PATH   = os.getenv(
    'EMB_PATH',
    './probs_2512_embs.npy'
)
PROB_PATH  = os.getenv(
    'PROB_PATH',
    './probs_2512.jsonl'
)
BF_16 = os.getenv(
    "BF_16",
    1,
)

PAGE_SIZE  = 20         # Number of results per page
# ------------------------------------- #

app = Flask(__name__)

# ---------- Load model & data on startup ---------- #
print("Loading SentenceTransformer model …")
if BF_16 == 1:
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})
else:
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.tokenizer.model_max_length = 2048
model.max_seq_length            = 2048

print("Loading pre‑computed embeddings …")
embs = np.load(EMB_PATH).astype("float32")
embs /= np.linalg.norm(embs, axis=1, keepdims=True)

print("Loading problem metadata …")
probs = [json.loads(line) for line in open(PROB_PATH, "r", encoding="utf‑8")]

# 收集所有可用的 OJ 源，用于前端下拉菜单
available_ojs = sorted(list(set(p.get("source") for p in probs if p.get("source"))))

assert len(probs) == embs.shape[0], "Mismatch between vector and problem count!"
print(f"Ready! {len(probs)} problems indexed.\n")


from functools import lru_cache
import hashlib

def _hash(text: str) -> str:
    """Simple hash to shorten long queries as dict keys"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

@lru_cache(maxsize=1024)   # Cache up to 1024 different queries
def search_once(q: str):
    """Return ranked indices and similarity list (numpy array -> Python list)"""
    # -> float32
    q_emb = model.encode(q, convert_to_tensor=True).to(torch.float32).cpu().numpy()
    q_emb = q_emb / np.linalg.norm(q_emb)
    sims  = embs.dot(q_emb)
    idx   = sims.argsort()[::-1]
    return idx.tolist(), sims.tolist()

def load_search_stats():
    """Load daily search statistics from file."""
    if not os.path.exists(SEARCH_STATS_PATH):
        return {}
    with open(SEARCH_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_search_stats(stats: dict):
    """Save search statistics to file."""
    with open(SEARCH_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def record_search():
    """Update today's search count and save to file."""
    stats = load_search_stats()
    today = datetime.now().strftime("%Y-%m-%d")
    stats[today] = stats.get(today, 0) + 1
    save_search_stats(stats)

@app.route("/", methods=["GET"])
def index():
    lang  = detect_lang()
    t     = I18N[lang]

    q     = request.args.get("q", "").strip()
    page  = max(int(request.args.get("page", "1")), 1)

    if "oj" in request.args:
        selected_ojs = request.args.getlist("oj")
    else:
        selected_ojs = available_ojs

    results, total, elapsed = [], 0, 0.0

    if q:
        record_search()
        tic   = time.perf_counter()
        idx, sims = search_once(q)
        elapsed = (time.perf_counter() - tic) * 1_000 

        # 根据 OJ 筛选结果
        filtered_idx = []
        for j in idx:
            p = probs[j]
            # 如果选中了特定的 OJ，并且当前问题的 source 不在选中列表中，则跳过
            if p.get("source") in selected_ojs:
                filtered_idx.append(j)
        
        total   = len(filtered_idx)

        results = []
        start, end = (page - 1) * PAGE_SIZE, page * PAGE_SIZE
        for rank, j in enumerate(filtered_idx[start:end], start=start + 1):
            p = probs[j]
            results.append({
                "rank"  : rank,
                "pid"   : j,
                "score" : float(sims[j]),
                "title" : p.get("title") or t["untitled"],
                "url"   : p.get("url", "#"),
                "source": p.get("source", ""),
            })


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
        available_ojs=available_ojs,
        selected_ojs=selected_ojs,
    )

@app.route("/p/<int:pid>")
def problem(pid: int):
    lang = detect_lang()
    t    = I18N[lang]

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
        query=request.args.get("q", ""),       # Pass original query to return button
        page=request.args.get("page", "1"),
        selected_ojs_str=request.args.get("oj", "") # Pass selected_ojs_str for back button
    )

@app.route("/stats")
def stats():
    lang = detect_lang()
    t    = I18N[lang]

    stats = load_search_stats()
    stats_data = sorted(stats.items(), key=lambda x: x[0], reverse=True)

    stats_draw = list(reversed(stats_data))
    stats_draw = stats_draw[:-1] # Exclude today

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