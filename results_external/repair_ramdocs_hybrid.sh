#!/bin/zsh
# The 2026-11-20 RAMDocs hybrid run lost 41 items to NVIDIA 429s caused by a runaway
# smoke-test process. Once that run has written its metrics, drop the fallback rows
# from its checkpoint and resume the run so only those items are regenerated.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
until ls $R/ramdocs_nvidia_hybrid_metrics_2026112*.json >/dev/null 2>&1; do sleep 60; done
CK=$R/ramdocs_nvidia_hybrid_nvidia-nemotron-3-5-super-vl-120b-a12b_checkpoint.jsonl
cp "$CK" "$CK.with_429_fallbacks.bak"
$PY - "$CK" <<'PYEOF'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
keep = [r for r in rows if r.get("reason") != "Generation failed; no answer was selected."]
open(p, "w").write("".join(json.dumps(r, default=str) + "\n" for r in keep))
print(f"checkpoint: kept {len(keep)} of {len(rows)} rows; {len(rows)-len(keep)} will be regenerated")
PYEOF
$PY evaluate_external.py --dataset ramdocs --mode hybrid --workers 2 --generation-model $M > $R/progress_ramdocs_hybrid_repair.out 2>&1
echo "HYBRID REPAIR DONE $(date)" > $R/progress_hybrid_repair_done.txt
