#!/bin/zsh
# GOV-RAG improvement variants on RAMDocs (Nemotron via NVIDIA API). Three workers,
# then a two-worker retry pass per variant. No judging; automatic metrics only.
#   V1 hybrid retrieval, strict policy        -> tag ramdocs_nvidia_govrag_hybrid_strict
#   V2 dense retrieval, answer-and-flag       -> tag ramdocs_nvidia_govrag_dense_answerandflag
#   V3 hybrid retrieval, answer-and-flag      -> tag ramdocs_nvidia_govrag_hybrid_answerandflag
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
rm -f $R/progress_govrag_variants_ramdocs_done.txt
for v in "hybrid strict" "dense answer_and_flag" "hybrid answer_and_flag"; do
  set -- ${=v}; ret=$1; pol=$2; name=${ret}_${pol//_/}
  $PY evaluate_external.py --dataset ramdocs --mode govrag --govrag-retrieval $ret --govrag-policy $pol --workers 3 --generation-model $M > $R/progress_ramdocs_govrag_${name}.out 2>&1
  $PY evaluate_external.py --dataset ramdocs --mode govrag --govrag-retrieval $ret --govrag-policy $pol --workers 2 --generation-model $M > $R/progress_ramdocs_govrag_${name}_retry.out 2>&1
done
echo "GOVRAG VARIANTS RAMDOCS DONE $(date)" > $R/progress_govrag_variants_ramdocs_done.txt
