# CPRet Local Retrieval Service

A Flask web service for searching competitive-programming problems with CPRet embeddings.
For the dataset, models, and training, see the [main README](../README.md).

## Directory Layout

```
cp-retrieval-server/
├── app.py              # Flask entry point (web UI + /api/search)
├── run.sh              # Unified command entry (start / status / build / reload ...)
├── encoder/            # Query encoders: vLLM (sync/async), sentence_transformers, mock
├── retrieval/          # Problem data loading, search context, result insights
├── admin/              # Hot updates: local Unix-socket admin, bundle preparation
├── scripts/            # download.py, compute_embs.py, example.py
├── templates/ static/  # Web front end
└── updates/            # Data bundles for hot updates (not in git)
```

## 1. Download Data

Download the problems and embeddings from [HF dataset CPRet-Embeddings](https://huggingface.co/datasets/coldchair16/CPRet-Embeddings) into `cp-retrieval-server/`:

```bash
cd cp-retrieval-server
python scripts/download.py
```

* New model `CPRetriever-Prob-Qwen3-4B-2510`: `probs_2611.jsonl`, `probs_2611_embs.npy`
* Old model `CPRetriever-Prob`: `probs.jsonl`, `probs_embs.npy`

## 2. Install

```bash
pip install -r ../requirements.txt
pip install -r requirements-vllm.txt   # Only needed for the vLLM backend
```

## 3. Start the Service

### Recommended: `run.sh` (vLLM)

```bash
bash run.sh start                          # async vLLM, 2048-token queries, 8192 engine capacity
MAX_QUERY_TOKENS=8192 bash run.sh start    # Use 8192-token queries from startup
VLLM_MODE=sync VLLM_MAX_NUM_SEQS=1 bash run.sh start   # Fall back to synchronous vLLM
bash run.sh start sentence_transformers    # Original sentence_transformers backend
HOST=127.0.0.1 PORT=5001 bash run.sh start
bash run.sh help                           # All commands
```

Run as a single process (threaded). Do not use multiple workers or preload.
`run.sh start` also enables the local admin socket (see section 4).

### Minimal: `python app.py`

```bash
python app.py     # sentence_transformers backend, default model CPRetriever-Prob-Qwen3-4B-2510
```

For the **old model (`CPRetriever-Prob`)**:

```bash
export MODEL_PATH=coldchair16/CPRetriever-Prob
export EMB_PATH=./probs_embs.npy
export PROB_PATH=./probs.jsonl
python app.py
```

bf16 is enabled by default; set `BF_16=0` if your device does not support it.

### Main Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` / `PROB_PATH` / `EMB_PATH` | new model / `probs_2611*` | Model and data |
| `ENCODER_BACKEND` | `sentence_transformers` (`vllm` in `run.sh`) | Encoder backend |
| `MAX_QUERY_TOKENS` | `2048` | Query truncation length (`2048` or `8192`) |
| `ENGINE_MAX_QUERY_TOKENS` | `8192` | vLLM engine capacity |
| `VLLM_MODE` | `async` | `async` (concurrent) or `sync` |
| `VLLM_MAX_NUM_SEQS` / `VLLM_MAX_NUM_BATCHED_TOKENS` | `2` / `2048` | vLLM concurrency and batch budget |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.8` | vLLM GPU memory fraction |
| `ENCODER_MAX_PENDING` / `ENCODER_TIMEOUT_SECONDS` | `32` / `120` | Queue limit (503 when full) and timeout (504) |
| `BF_16` | `1` | bf16 inference |
| `HOST` / `PORT` | `0.0.0.0` / `5000` | Listen address |
| `FLASK_SECRET_KEY` | random per start | Session key; set it to keep query IDs valid across restarts |
| `MOCK_MODEL` | `0` | Debug mode without loading the model |

## 4. Hot Updates (no restart)

Management is only available through a local Unix socket (`.runtime/admin.sock`, mode 0700); there is no HTTP admin endpoint.

```bash
bash run.sh status                                    # Current data, query length, encoder, build progress
bash run.sh length 8192                               # Switch query length only
bash run.sh build updates/probs_new.jsonl 2026-12 8192   # Build a bundle in the background with the running GPU model
watch -n 2 'bash run.sh status'                       # Wait for build.state = ready
bash run.sh reload updates/2026-12 8192               # Activate the bundle
bash run.sh rollback                                  # Roll back the last switch
bash run.sh cancel-build
```

* The new JSONL must contain all existing problems; existing row order is preserved.
* Input files and bundles must be inside `updates/` (`DATA_UPDATE_ROOT`).
* Compatible short problems and exact duplicates reuse existing vectors; only the rest are recomputed.
* The active bundle is stored in `.runtime/active.json` and restored on the next start.
* Offline preparation: `bash run.sh prepare ...` and `bash run.sh compute ...` (see `bash run.sh help`).

## 5. Search API

* Query text is sent by POST; the URL only carries a short query ID (`qid`) for paging and problem links.
* A `qid` is bound to the originating browser session and expires after 1 hour or on restart.

```bash
curl -X POST http://127.0.0.1:5000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"q": "shortest path with exactly k edges", "page": 1}'
```

Legacy `GET /api/search?q=...` is still accepted for compatibility.

## 6. About the Dataset

The current problem database (collected up to **November 2026**) includes problems from:
[Codeforces](https://codeforces.com/), [Codeforces Gym](https://codeforces.com/gyms), [AtCoder](https://atcoder.jp/),
[Leetcode](https://leetcode.com/), [SPOJ](https://www.spoj.com/), [Nowcoder](https://ac.nowcoder.com/),
[Luogu](https://www.luogu.com.cn/), [Loj](https://loj.ac/), [CodeChef](https://www.codechef.com/dashboard),
[AIZU](https://judge.u-aizu.ac.jp/onlinejudge/), [UOJ](https://uoj.ac/), [QOJ](https://qoj.ac/), [Hydro](https://hydro.ac/).

You can add your own data source and generate embeddings with [`scripts/compute_embs.py`](scripts/compute_embs.py)
(about 4 GPU hours on an H100 for the current database), or use the incremental `build` above.

If you have access to a larger or more diverse problem dataset, **we welcome contributions** —
contact us (2317757009@qq.com) or open an issue/pull request.

## 7. System Requirements

The service runs on **CPU** or **GPU**:

* **2B old model** (`CPRetriever-Prob`): at least **16GB** of memory or VRAM.
* **4B new model** (`CPRetriever-Prob-Qwen3-4B-2510`): **32GB or more** (fp32); with bf16, half of that.

Typical query latency: **10–20 s** on CPU (8 cores), **0.1–1 s** on GPU (e.g., A800), depending on input length.
