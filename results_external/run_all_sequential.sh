#!/bin/zsh
# Runs the four external evaluations one after another, resuming from checkpoints.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
$PY evaluate_external.py --dataset ramdocs   --mode govrag   --workers 3 --generation-model $M > $R/progress_ramdocs_govrag.out 2>&1
$PY evaluate_external.py --dataset ramdocs   --mode baseline --workers 3 --generation-model $M > $R/progress_ramdocs_baseline.out 2>&1
$PY evaluate_external.py --dataset conflicts --mode govrag   --workers 3 --judge --generation-model $M > $R/progress_conflicts_govrag.out 2>&1
$PY evaluate_external.py --dataset conflicts --mode baseline --workers 3 --judge --generation-model $M > $R/progress_conflicts_baseline.out 2>&1
echo "ALL DONE $(date)" > $R/progress_all_done.txt
