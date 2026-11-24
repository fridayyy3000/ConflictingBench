#!/bin/zsh
# Starts both GOV-RAG variant chains detached from this window (Terminal.app owns them,
# so quitting or updating Claude does not touch them). Safe to close this window after.
cd "$(dirname "$0")" || exit 1
nohup caffeinate -i ./run_govrag_variants_ramdocs_resume.sh > terminal_ramdocs.out 2>&1 &
nohup caffeinate -i ./run_govrag_variants_conflicts.sh > terminal_conflicts.out 2>&1 &
sleep 5
echo; pgrep -fl "run_govrag_variants|evaluate_external" 
echo; echo "Both chains started $(date). You may close this window; do NOT quit Terminal."
