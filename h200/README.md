# H200 folder: judging and the remaining runs

This folder is copied as a whole to `/mnt/data1/gnajpand/conflictbench_h200/` on the H200.
It is self-contained: no git clone, no NVIDIA key, no re-download of the datasets. The only
secret-free thing the judge needs is `OPENAI_API_KEY=EMPTY`, which the scripts set.

What is inside after `make_bundle.sh` has run on the Mac:

| Path | What | Why |
|---|---|---|
| `corvic/` | the harness code (`judge_three_way.py`, `build_judge_table.py`, `evaluate_external.py`, `pipeline/`) | judging imports the harness |
| `data/external/` | RAMDocs and CONFLICTS raw files, the exact local copies the results came from | gold answers and conflict types |
| `data/embeddings/` | bge-small embedding cache | only if answers are ever regenerated |
| `results_external/` | the newest `*_details_*.csv` of every run tag (judge input), every `*_metrics_*.json`, and the GLM judge outputs as the consistency reference | what gets judged, and what to compare the new judge against |
| `research_notes/H200_HANDOFF.md` | the decisions and the full plan | context |
| `setup.sh`, `start_judge.sh`, `judge_all.sh`, `compare_with_glm.py` | the run scripts | see below |

## On the Mac: build and copy the folder

```bash
bash h200/make_bundle.sh
```

```bash
rsync -av --progress h200_bundle/ h200:/mnt/data1/gnajpand/conflictbench_h200/
```

Re-run both whenever new results land on the Mac (the CONFLICTS variant rows); rsync only
sends what changed.

## On the H200, in order

Run everything inside `tmux` so a dropped SSH session does not kill it.

```bash
cd /mnt/data1/gnajpand/conflictbench_h200
```

1. `bash setup.sh` creates two virtual environments, `.venv` for the harness and
   `.venv-vllm` for the judge server, so vLLM's pinned torch cannot fight the harness.
   `SKIP_VLLM=1 bash setup.sh` skips the vLLM install if one already exists on the machine.
2. `bash start_judge.sh` starts vLLM on port 8000 with `Qwen/Qwen3-32B` (override with
   `JUDGE_MODEL=openai/gpt-oss-120b`), downloads the weights into `hf_cache/` here on
   `/mnt/data1`, and waits until the server answers. Logs: `logs/vllm_*.log`.
3. `bash judge_all.sh --smoke` judges 20 rows of the system's RAMDocs run. Read a few
   verdicts in `results_external/*_judge3_*_checkpoint.jsonl` before going on.
4. `bash judge_all.sh` judges the newest details CSV of every run tag, resumes from
   checkpoints if interrupted, then builds `results_external/judge3_table_<slug>.md` and
   prints the comparison against the GLM verdicts. `judge_all.sh --list` only prints what
   would be judged. `SKIP_TAGS="ramdocs_nvidia_openwebui" bash judge_all.sh` skips rows.
5. Copy the verdicts back to the Mac and commit them from there:

```bash
rsync -av h200:/mnt/data1/gnajpand/conflictbench_h200/results_external/ ~/Documents/Conflicting_documents/results_external/ --include='*_judge3_*' --include='judge3_table_*' --exclude='*'
```

## Judge sanity check

`compare_with_glm.py` prints, for every row both judges graded, FC and Inc from GLM next to
the new judge, and whether the two judges rank the methods the same way within each dataset.
FC and Inc should move together across methods; one method moving a lot on its own means the
new judge reads abstentions or hedges differently and the prompt needs a look before the
table is used.
