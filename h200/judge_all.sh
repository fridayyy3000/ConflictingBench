#!/usr/bin/env bash
# Judge every run with the self-hosted model, then build the table and compare with GLM.
#   bash judge_all.sh --list    print the run tags and CSVs that would be judged
#   bash judge_all.sh --smoke   judge 20 rows of the system's RAMDocs run
#   bash judge_all.sh           judge everything (resumes from checkpoints)
# Env: JUDGE_MODEL (default Qwen/Qwen3-32B), JUDGE_PORT (8000), JUDGE_WORKERS (16),
#      SKIP_TAGS (space-separated run tags to skip, e.g. "ramdocs_nvidia_openwebui").
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
PORT="${JUDGE_PORT:-8000}"
WORKERS="${JUDGE_WORKERS:-16}"
SKIP_TAGS="${SKIP_TAGS:-}"
SLUG="$(printf '%s' "$MODEL" | sed -E 's/[^A-Za-z0-9]+/-/g; s/^-//; s/-$//')"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
J=(--judge-model "$MODEL" --base-url "http://localhost:$PORT/v1" --workers "$WORKERS")
mkdir -p logs

R="$ROOT/results_external"
newest_for() { ls -t "$R"/"$1"_details_*.csv 2>/dev/null | head -1; }
TAGS=$(ls "$R"/*_details_*.csv | sed -E 's|.*/||; s/_details_[0-9_]+\.csv$//' | sort -u)

if [ "${1:-}" = "--list" ]; then
  for tag in $TAGS; do
    case " $SKIP_TAGS " in *" $tag "*) echo "skip   $tag"; continue;; esac
    f=$(newest_for "$tag"); echo "judge  $tag  <- $(basename "$f")  ($(( $(wc -l < "$f") - 1 )) rows)"
  done
  echo "judge model: $MODEL  (slug $SLUG), $WORKERS workers, port $PORT"
  exit 0
fi

if ! curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
  echo "No judge server on port $PORT. Run: bash start_judge.sh" >&2
  exit 1
fi

cd corvic
if [ "${1:-}" = "--smoke" ]; then
  f=$(newest_for ramdocs_nvidia_govrag_hybrid_answerandflag)
  echo "== smoke: 20 rows of $(basename "$f")"
  "$PY" judge_three_way.py "$f" "${J[@]}" --limit 20
  echo "== read a few verdicts:"
  tail -5 "$R"/ramdocs_nvidia_govrag_hybrid_answerandflag_judge3_"$SLUG"_checkpoint.jsonl | cut -c1-300
  exit $?
fi

failed=()
for tag in $TAGS; do
  case " $SKIP_TAGS " in *" $tag "*) echo "== skip $tag"; continue;; esac
  f=$(newest_for "$tag")
  echo "== $tag  ($(basename "$f"))  $(date)"
  if ! "$PY" judge_three_way.py "$f" "${J[@]}" 2>&1 | tee -a "../logs/judge_${tag}.log"; then
    failed+=("$tag")
  fi
done

echo "== table"
"$PY" build_judge_table.py --judge-slug "$SLUG" || failed+=("build_judge_table")
echo "== comparison with GLM"
"$PY" ../compare_with_glm.py --results "$R" --slug "$SLUG" || true

if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED: ${failed[*]}  (rerun judge_all.sh; it resumes from the checkpoints)" >&2
  exit 1
fi
echo "== all rows judged with $MODEL at $(date). Table: results_external/judge3_table_$SLUG.md"
