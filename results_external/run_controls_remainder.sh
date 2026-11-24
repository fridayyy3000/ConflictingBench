#!/bin/zsh
# Remainder of the controls chain after the 20 Nov stall: CONFLICTS closed-book (resumes
# from its 138-row checkpoint), CONFLICTS hybrid, then GLM judging of the four control
# runs. Writes the same done marker the LlamaIndex chain is waiting for.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
rm -f $R/progress_controls_done.txt
$PY evaluate_external.py --dataset conflicts --mode closedbook --workers 3 --generation-model $M > $R/progress_conflicts_closedbook.out 2>&1
# second pass retries anything that failed on transport (not checkpointed); no-op if none
$PY evaluate_external.py --dataset conflicts --mode closedbook --workers 2 --generation-model $M > $R/progress_conflicts_closedbook_retry.out 2>&1
$PY evaluate_external.py --dataset conflicts --mode hybrid --workers 3 --generation-model $M > $R/progress_conflicts_hybrid.out 2>&1
$PY evaluate_external.py --dataset conflicts --mode hybrid --workers 2 --generation-model $M > $R/progress_conflicts_hybrid_retry.out 2>&1
FILES=()
for ds in ramdocs conflicts; do for mode in closedbook hybrid; do
  f=$(ls -t $R/${ds}_nvidia_${mode}_details_*.csv 2>/dev/null | head -1); [ -n "$f" ] && FILES+=("$f")
done; done
$PY judge_three_way.py "${FILES[@]}" --judge-model z-ai/glm-5.3-flash --workers 6 --max-tokens 4096 --out-dir $R > $R/progress_judge3_glm_controls.out 2>&1
echo "CONTROLS DONE $(date)" > $R/progress_controls_done.txt
