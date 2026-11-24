#!/usr/bin/env bash
# Start the judge model with vLLM in the background and wait until it answers.
#   JUDGE_MODEL  default Qwen/Qwen3-32B   (alternative: openai/gpt-oss-120b)
#   JUDGE_PORT   default 8000
# Weights download into ./hf_cache (on /mnt/data1, not the home directory).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
PORT="${JUDGE_PORT:-8000}"
export HF_HOME="$PWD/hf_cache"
mkdir -p logs

if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo "A server already answers on port $PORT:"
  curl -fsS "http://localhost:$PORT/v1/models" | python3 -c "import json,sys; print('  ', [m['id'] for m in json.load(sys.stdin)['data']])"
  exit 0
fi

if [ -x .venv-vllm/bin/vllm ]; then VLLM=.venv-vllm/bin/vllm; else VLLM="$(command -v vllm)"; fi
LOG="logs/vllm_$(date +%Y%m%d_%H%M%S).log"
echo "== starting $MODEL on port $PORT with $VLLM; log: $LOG"
nohup "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  > "$LOG" 2>&1 &
echo $! > logs/vllm.pid
echo "   pid $(cat logs/vllm.pid)"

echo "== waiting for the server (first start downloads the weights; can take a while)"
for i in $(seq 1 360); do
  if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "   ready after ~$((i * 10)) s"
    curl -fsS "http://localhost:$PORT/v1/models" | python3 -c "import json,sys; print('   serving', [m['id'] for m in json.load(sys.stdin)['data']])"
    echo "== next: bash judge_all.sh --smoke"
    exit 0
  fi
  if ! kill -0 "$(cat logs/vllm.pid)" 2>/dev/null; then
    echo "vLLM exited; last log lines:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 10
done
echo "Server not ready after 60 minutes; check $LOG" >&2
exit 1
