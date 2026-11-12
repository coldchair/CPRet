#!/usr/bin/env bash
set -euo pipefail
umask 077

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$APP_DIR"

usage() {
  cat <<'HELP'
用法：bash run.sh <命令> [参数]

启动（单进程多线程；不要使用多 worker 或 preload）：
  bash run.sh start                         # 默认异步 vLLM，2 条运行序列；2048 查询/8192 容量
  VLLM_MODE=sync VLLM_MAX_NUM_SEQS=1 bash run.sh start  # 回退原同步 vLLM
  VLLM_MAX_NUM_SEQS=4 bash run.sh start      # 显存验证后再尝试更高并发
  bash run.sh start sentence_transformers   # 保留旧推理方式
  MAX_QUERY_TOKENS=8192 bash run.sh start    # 启动即使用 8192
  HOST=127.0.0.1 PORT=5001 bash run.sh start # 指定地址、端口
  Ctrl+C                                   # 结束前台服务

本地管理（必须由服务所属用户执行；只有 Unix socket，没有 HTTP 管理接口）：
  bash run.sh status
  bash run.sh length 8192                   # 只切查询长度，不重算旧题库
  bash run.sh length 2048
  bash run.sh build updates/probs_new.jsonl 2026-12 8192
  bash run.sh status                        # 等待 build.state=ready
  bash run.sh reload updates/2026-12 8192   # 切换题库及查询长度
  bash run.sh rollback                      # 回滚上一次切换
  bash run.sh cancel-build

  bash run.sh build updates/probs_2611.jsonl 2026-11 8192

外机生成/导入（prepare 不会自动切换网站）：
  bash run.sh prepare --input updates/probs_new.jsonl --output updates/2026-12 \
    --max-length 8192 --trust-legacy-embeddings \
    --import-probs updates/delta.jsonl --import-embs updates/delta.npy \
    --import-manifest updates/delta-manifest.json
  bash run.sh prepare --input updates/probs_new.jsonl --output updates/2026-12 \
    --base-bundle updates/2026-11 --max-length 8192 --compute-missing
  bash run.sh compute --model_path ./CPRetriever-Prob-Qwen3-4B-2510 \
    --input_path updates/delta.jsonl --max_length 8192 \
    --manifest_path updates/delta-manifest.json

评估：
  bash run.sh evaluate-context
  bash run.sh evaluate-relevance
  bash run.sh test                          # tests/ 与 local/ 的单元测试

说明：
  新 JSONL 必须包含已有题目；构建保留旧行号，缺失旧题会拒绝发布。
  从 2048 升级题库到 8192 时，旧长题也要重算；兼容的短题及完全重复题复用向量。
  在线重算请使用 build，共享当前 GPU 模型；不要同时另启 compute/prepare --compute-missing。
  prepare 默认不加载 GPU；如果导入仍缺少兼容向量，会明确失败，不混用不同长度的长题向量。
  updates 与 socket 父目录要求本用户独占、权限 0700；版本目录发布后不要原地修改。
  查询正文走 POST；短查询 ID 仅对原浏览器有效，有效期 1 小时，可因容量淘汰或重启失效。
  QUERY_LOG=0 关闭本地查询记录（local/query_log.py 存在时默认开启，写入 query_logs/）。
  MODEL_PATH、EMB_PATH、PROB_PATH、LOCAL_ADMIN_SOCKET、DATA_UPDATE_ROOT、PYTHON 可用环境变量覆盖。
HELP
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$command" == "help" || "$command" == "--help" || "$command" == "-h" ]]; then
  usage
  exit 0
fi

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
    source /usr/local/miniconda3/etc/profile.d/conda.sh
    conda activate base
  fi
  PYTHON="$(command -v python)"
fi
export MODEL_PATH="${MODEL_PATH:-$APP_DIR/CPRetriever-Prob-Qwen3-4B-2510}"
export EMB_PATH="${EMB_PATH:-$APP_DIR/probs_2611_embs.npy}"
export PROB_PATH="${PROB_PATH:-$APP_DIR/probs_2611.jsonl}"
export LOCAL_ADMIN_SOCKET="${LOCAL_ADMIN_SOCKET:-$APP_DIR/.runtime/admin.sock}"
export DATA_UPDATE_ROOT="${DATA_UPDATE_ROOT:-$APP_DIR/updates}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

require_dir() {
  [[ -d "$APP_DIR/$1" ]] || { echo "$1/ 目录不存在（本地开发目录，不在 git 中）" >&2; exit 2; }
}

case "$command" in
  start)
    [[ $# -le 1 ]] || { usage; exit 2; }
    export ENCODER_BACKEND="${1:-${ENCODER_BACKEND:-vllm}}"
    export LOCAL_ADMIN=1
    export MAX_QUERY_TOKENS="${MAX_QUERY_TOKENS:-2048}"
    export ENGINE_MAX_QUERY_TOKENS="${ENGINE_MAX_QUERY_TOKENS:-8192}"
    export VLLM_MODE="${VLLM_MODE:-async}"
    export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
    export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
    export VLLM_LONG_PREFILL_THRESHOLD="${VLLM_LONG_PREFILL_THRESHOLD:-512}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    if [[ -f "$APP_DIR/local/query_log.py" && "${QUERY_LOG:-1}" != "0" ]]; then
      exec "$PYTHON" -m local.query_log
    fi
    exec "$PYTHON" "$APP_DIR/app.py"
    ;;
  status|rollback|cancel-build)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    exec "$PYTHON" -m admin.local_admin --socket "$LOCAL_ADMIN_SOCKET" "$command"
    ;;
  length)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    exec "$PYTHON" -m admin.local_admin --socket "$LOCAL_ADMIN_SOCKET" set-query-length --tokens "$1"
    ;;
  build)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    exec "$PYTHON" -m admin.local_admin --socket "$LOCAL_ADMIN_SOCKET" build --input "$1" --version "$2" --tokens "${3:-8192}"
    ;;
  reload)
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    args=(--socket "$LOCAL_ADMIN_SOCKET" reload --bundle "$1")
    if [[ $# -eq 2 ]]; then args+=(--query-tokens "$2"); fi
    exec "$PYTHON" -m admin.local_admin "${args[@]}"
    ;;
  prepare)
    exec "$PYTHON" -m admin.prepare_update --model-path "$MODEL_PATH" --base-probs "$PROB_PATH" --base-embs "$EMB_PATH" "$@"
    ;;
  compute)
    exec "$PYTHON" -m scripts.compute_embs "$@"
    ;;
  evaluate-context)
    require_dir benchmarks
    exec "$PYTHON" -m benchmarks.evaluate_context_length --model-path "$MODEL_PATH" --prob-path "$PROB_PATH" --emb-path "$EMB_PATH" "$@"
    ;;
  evaluate-relevance)
    require_dir benchmarks
    exec "$PYTHON" -m benchmarks.evaluate_relevance --model-path "$MODEL_PATH" --prob-path "$PROB_PATH" --emb-path "$EMB_PATH" "$@"
    ;;
  test)
    require_dir tests
    targets=(tests)
    [[ -d local ]] && targets+=(local)
    for target in "${targets[@]}"; do "$PYTHON" -m unittest discover -v -s "$target" -t . -p 'test_*.py'; done
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
