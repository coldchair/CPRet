import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from sentence_transformers import SentenceTransformer
import time
from flask import Flask, request, render_template, redirect, url_for
from markupsafe import Markup, escape

# ===== Multilingual dictionary =====
I18N = {
    "zh": {
        "site_name" : "题目检索",
        "placeholder": "输入题目描述或简略题意 …",
        "template_btn": "填入示例查询",
        "search_btn": "搜索",
        "summary"   : "共 <strong>{total}</strong> 条结果，页 {page}/{max_page}，耗时 {elapsed:.1f} ms",
        "prev"      : "上一页",
        "next"      : "下一页",
        "untitled"  : "未命名",
        "view_origin": "原站链接",
        "back": "返回搜索",
    },
    "en": {
        "site_name" : "Problem Search",
        "placeholder": "Enter problem description or simplified statement…",
        "template_btn": "Insert example query",
        "search_btn": "Search",
        "summary"   : "<strong>{total}</strong> results, page {page}/{max_page}, {elapsed:.1f} ms",
        "prev"      : "Prev",
        "next"      : "Next",
        "untitled"  : "Untitled",
        "view_origin": "Original Link",
        "back": "Back to Search",
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
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "coldchair16/CPRetriever-Prob"
)
EMB_PATH   = "./probs_embs.npy"
PROB_PATH  = "./probs.jsonl"
PAGE_SIZE  = 20         # Number of results per page
# ------------------------------------- #

app = Flask(__name__)

# ---------- Load model & data on startup ---------- #
print("Loading SentenceTransformer model …")
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.tokenizer.model_max_length = 1024
model.max_seq_length            = 1024

print("Loading pre‑computed embeddings …")
embs = np.load(EMB_PATH).astype("float32")
embs /= np.linalg.norm(embs, axis=1, keepdims=True)

print("Loading problem metadata …")
probs = [json.loads(line) for line in open(PROB_PATH, "r", encoding="utf‑8")]

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
    q_emb = model.encode(q, convert_to_tensor=True).cpu().numpy()
    q_emb = q_emb / np.linalg.norm(q_emb)
    sims  = embs.dot(q_emb)
    idx   = sims.argsort()[::-1]
    return idx.tolist(), sims.tolist()

@app.route("/", methods=["GET"])
def index():
    lang  = detect_lang()
    t     = I18N[lang]

    q     = request.args.get("q", "").strip()
    page  = max(int(request.args.get("page", "1")), 1)

    results, total, elapsed = [], 0, 0.0

    if q:
        tic   = time.perf_counter()
        idx, sims = search_once(q)
        elapsed = (time.perf_counter() - tic) * 1_000 
        total   = len(idx)

        results = []
        start, end = (page - 1) * PAGE_SIZE, page * PAGE_SIZE
        for rank, j in enumerate(idx[start:end], start=start + 1):
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
    )

# -------------- Local run entry -------------- #
if __name__ == "__main__":
    # export FLASK_ENV=development to enable auto-reload/debug
    app.run(host="0.0.0.0", port=5000, debug=False)