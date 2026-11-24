#!/bin/zsh
# Resume of run_govrag_variants_ramdocs.sh after the app relaunch killed it.
# V1 (hybrid strict) is complete. V2 resumes from its checkpoint, then V3 runs.
# Three workers, then a two-worker retry pass. No judging.
cd "$(dirname "$0")/../corvic" || exit 1
PY=/opt/anaconda3/bin/python3
M=nvidia/nemotron-3.5-super-vl-120b-a12b
R=../results_external
export EMBED_DEVICE=cpu
rm -f $R/progress_govrag_variants_ramdocs_done.txt
for v in "dense answer_and_flag" "hybrid answer_and_flag"; do
  set -- ${=v}; ret=$1; pol=$2; name=${ret}_${pol//_/}
  $PY evaluate_external.py --dataset ramdocs --mode govrag --govrag-retrieval $ret --govrag-policy $pol --workers 3 --generation-model $M >> $R/progress_ramdocs_govrag_${name}.out 2>&1
  $PY evaluate_external.py --dataset ramdocs --mode govrag --govrag-retrieval $ret --govrag-policy $pol --workers 2 --generation-model $M > $R/progress_ramdocs_govrag_${name}_retry.out 2>&1
done
echo "GOVRAG VARIANTS RAMDOCS DONE $(date)" > $R/progress_govrag_variants_ramdocs_done.txt
