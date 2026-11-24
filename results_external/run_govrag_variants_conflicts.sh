#!/bin/zsh
# GOV-RAG improvement variants on CONFLICTS (Nemotron via NVIDIA API). Waits for the
# RAMDocs variant chain to finish, then runs each variant with three workers and a
# two-worker retry pass. No judging; automatic metrics only. Most important variant first.
#   V3 hybrid retrieval, answer-and-flag -> tag conflicts_nvidia_govrag_hybrid_answerandflag
#   V1 hybrid retrieval, strict policy   -> tag conflicts_nvidia_govrag_hybrid_strict
#   V2 dense retrieval, answer-and-flag  -> tag conflicts_nvidia_govrag_dense_answerandflag
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
rm -f $R/progress_govrag_variants_conflicts_done.txt
while [ ! -f $R/progress_govrag_variants_ramdocs_done.txt ]; do
  pgrep -f run_govrag_variants_ramdocs >/dev/null || break
  sleep 60
done
echo "CONFLICTS CHAIN STARTED $(date)" > $R/progress_govrag_variants_conflicts_started.txt
for v in "hybrid answer_and_flag" "hybrid strict" "dense answer_and_flag"; do
  set -- ${=v}; ret=$1; pol=$2; name=${ret}_${pol//_/}
  $PY evaluate_external.py --dataset conflicts --mode govrag --govrag-retrieval $ret --govrag-policy $pol --workers 3 --generation-model $M > $R/progress_conflicts_govrag_${name}.out 2>&1
  $PY evaluate_external.py --dataset conflicts --mode govrag --govrag-retrieval $ret --govrag-policy $pol --workers 2 --generation-model $M > $R/progress_conflicts_govrag_${name}_retry.out 2>&1
done
echo "GOVRAG VARIANTS CONFLICTS DONE $(date)" > $R/progress_govrag_variants_conflicts_done.txt
