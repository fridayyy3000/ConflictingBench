#!/bin/zsh
# Waits for the GLM judge pass to finish, then runs the two new Nemotron
# controls (closed-book few-shot CoT, hybrid BM25+dense RAG) on both datasets,
# one process at a time, then judges the four new runs with GLM-5.3-Flash.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
until [ -f $R/progress_judge3_done.txt ]; do sleep 60; done
rm -f $R/progress_controls_done.txt
for ds in ramdocs conflicts; do
  for mode in closedbook hybrid; do
    $PY evaluate_external.py --dataset $ds --mode $mode --workers 3 --generation-model $M > $R/progress_${ds}_${mode}.out 2>&1
  done
done
# judge the four new runs (latest details CSV per tag)
FILES=()
for ds in ramdocs conflicts; do for mode in closedbook hybrid; do
  f=$(ls -t $R/${ds}_nvidia_${mode}_details_*.csv 2>/dev/null | head -1); [ -n "$f" ] && FILES+=("$f")
done; done
$PY judge_three_way.py "${FILES[@]}" --judge-model z-ai/glm-5.3-flash --workers 6 --max-tokens 4096 --out-dir $R > $R/progress_judge3_glm_controls.out 2>&1
echo "CONTROLS DONE $(date)" > $R/progress_controls_done.txt
