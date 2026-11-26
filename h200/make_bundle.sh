#!/usr/bin/env bash
# Assemble the self-contained H200 folder on the Mac (default: ../h200_bundle).
# Safe to re-run: code and data are mirrored, result files are refreshed to the newest.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO/h200_bundle}"
mkdir -p "$OUT"

echo "== scripts"
rsync -a --delete --exclude make_bundle.sh "$REPO/h200/" "$OUT/"

echo "== harness code (no corpora, static UI, example pack, caches or .env)"
rsync -a --delete \
  --exclude corpora --exclude static --exclude easy \
  --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache \
  --exclude '.env' --exclude '.env.*' \
  "$REPO/corvic/" "$OUT/corvic/"

echo "== datasets (exact local copies)"
rsync -a --delete "$REPO/data/external/" "$OUT/data/external/"
mkdir -p "$OUT/data/embeddings"
rsync -a "$REPO/data/embeddings/BAAI-bge-small-en-v1-5.sqlite" "$OUT/data/embeddings/"

echo "== handoff note"
mkdir -p "$OUT/research_notes"
rsync -a "$REPO/research_notes/H200_HANDOFF.md" "$OUT/research_notes/"

echo "== results: newest details CSV per run tag, metrics, GLM judge reference"
R="$REPO/results_external"
RO="$OUT/results_external"
mkdir -p "$RO"
rm -f "$RO"/*_details_*.csv
for tag in $(ls "$R"/*_details_*.csv | sed -E 's|.*/||; s/_details_[0-9_]+\.csv$//' | sort -u); do
  newest=$(ls -t "$R"/"${tag}"_details_*.csv | head -1)
  cp "$newest" "$RO/"
  echo "   $tag  <- $(basename "$newest")"
done
cp "$R"/*_metrics_*.json "$RO/"
cp "$R"/*_judge3_z-ai-glm-5-3-flash_*.json "$R"/*_judge3_z-ai-glm-5-3-flash_*.csv "$RO/" 2>/dev/null || true
cp "$R"/judge3_table_z-ai-glm-5-3-flash.md "$RO/" 2>/dev/null || true

mkdir -p "$OUT/logs" "$OUT/hf_cache"

if find "$OUT" -name '.env' -o -name '.env.*' | grep -q .; then
  echo "ERROR: a .env file ended up in the bundle; remove it before copying" >&2
  exit 1
fi

echo
echo "Bundle ready: $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "Copy it with:"
echo "  rsync -av --progress \"$OUT/\" h200:/mnt/data1/gnajpand/conflictbench_h200/"
