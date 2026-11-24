#!/bin/zsh
# Finishes the Rankify (monoT5 + basic-rag) CONFLICTS run from its checkpoint at two
# workers, then a retry pass for anything the rate limiter defeated. No judging.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
rm -f $R/progress_rankify_conflicts_done.txt
$PY evaluate_external.py --dataset conflicts --mode rankify --workers 2 --generation-model $M > $R/progress_conflicts_rankify.out 2>&1
$PY evaluate_external.py --dataset conflicts --mode rankify --workers 2 --generation-model $M > $R/progress_conflicts_rankify_retry.out 2>&1
echo "RANKIFY CONFLICTS DONE $(date)" > $R/progress_rankify_conflicts_done.txt
