#!/usr/bin/env bash
# One-time setup on the H200. Two virtual environments:
#   .venv       harness + judge client (judge_three_way.py, build_judge_table.py)
#   .venv-vllm  the judge server (vLLM pins its own torch; keep it apart)
# SKIP_VLLM=1 skips the vLLM environment (use when a working vllm is already on PATH).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "== python: $($PY --version)"

echo "== harness environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q numpy==1.26.4 openai sentence-transformers rank-bm25 pypdf python-docx requests tqdm
.venv/bin/python - <<'EOF'
import sys
sys.path.insert(0, "corvic")
import judge_three_way, build_judge_table  # noqa: F401  (imports the harness too)
print("   harness imports OK")
EOF

if [ "${SKIP_VLLM:-0}" = "1" ]; then
  echo "== vLLM environment skipped (SKIP_VLLM=1)"
else
  echo "== judge server environment (.venv-vllm); this downloads a CUDA torch, several GB"
  [ -d .venv-vllm ] || "$PY" -m venv .venv-vllm
  .venv-vllm/bin/pip install -q -U pip
  .venv-vllm/bin/pip install -q vllm
  .venv-vllm/bin/python -c "import vllm; print('   vllm', vllm.__version__)"
fi

mkdir -p logs hf_cache
echo "== done. Next: bash start_judge.sh"
