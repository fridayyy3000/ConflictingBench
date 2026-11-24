#!/bin/zsh
# Runs the LlamaIndex default-pipeline baseline (top-k 2 and 8) on both datasets
# once the closed-book / hybrid controls chain has finished, then GLM-judges them.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
until [ -f $R/progress_controls_done.txt ]; do sleep 60; done
rm -f $R/progress_llamaindex_done.txt
for ds in ramdocs conflicts; do
  for mode in llamaindex_k2 llamaindex_k8; do
    $PY evaluate_external.py --dataset $ds --mode $mode --workers 3 --generation-model $M > $R/progress_${ds}_${mode}.out 2>&1
  done
done
FILES=()
for ds in ramdocs conflicts; do for mode in llamaindex_k2 llamaindex_k8; do
  f=$(ls -t $R/${ds}_nvidia_${mode}_details_*.csv 2>/dev/null | head -1); [ -n "$f" ] && FILES+=("$f")
done; done
$PY judge_three_way.py "${FILES[@]}" --judge-model z-ai/glm-5.3-flash --workers 6 --max-tokens 4096 --out-dir $R > $R/progress_judge3_glm_llamaindex.out 2>&1
echo "LLAMAINDEX DONE $(date)" > $R/progress_llamaindex_done.txt
