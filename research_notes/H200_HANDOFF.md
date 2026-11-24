# H200 handoff — judging and remaining runs (written 23 Nov 2026, evening)

Deadline: NAACL 2027 System Demonstrations, 4 Dec 2026. Branch: `eval/baseline-comparison`
(pushed to `origin`). Generator for every row: `nvidia/nemotron-3.5-super-vl-120b-a12b` via
the NVIDIA API. Nothing in this note requires regenerating answers; the H200's job is judging.

## Decisions already made

| Decision | Value |
|---|---|
| System configuration ("Variant 3") | GOV-RAG with `--govrag-retrieval hybrid --govrag-policy answer_and_flag`, run tag `*_nvidia_govrag_hybrid_answerandflag`. Standalone copy with these defaults: `paper_pipeline/docuinsight/`. |
| Ablation rows | `*_govrag_dense_answerandflag` (policy only), `*_govrag_hybrid_strict` (retrieval only), `*_govrag` (original refuse-on-conflict GOV-RAG) |
| Judge for the paper table | **An open-weight model self-hosted on the H200, not GLM.** Every row is judged once by that model, including rows GLM already judged. GLM verdicts stay as a consistency reference only. |
| Metric columns | FC / PC / Inc (three-way, SentryLine convention); goal is highest FC and lowest Inc on both datasets |
| Not in the table | ConflictBench fictional pack (demo/walkthrough only), Open WebUI unless completed |
| Paper framing | Lead with conflict-aware answering (answer + flagged dissent + corroboration + verified citations); governance is a system feature, not a measured claim (benchmarks carry no governance metadata); enterprise is the motivation; rename away from "GOV-RAG" (name undecided) |

## What exists

Automatic metrics, Nemotron, Mac runs. RAMDocs is set exact match over 500 items; CONFLICTS
is answer containment over 458 items.

| Row (run tag suffix) | RAMDocs set-EM | RAMDocs misinfo-in-answer | CONFLICTS containment |
|---|---|---|---|
| govrag_hybrid_answerandflag (system) | 45.6 | 23.8 | 74.7 (8 unresolved, 1 invalid) |
| govrag_dense_answerandflag | 43.6 | 23.0 | 74.3 (10 unresolved, 0 invalid) |
| govrag_hybrid_strict | 37.6 | 36.6 | 76.0 (81 unresolved; containment counts refusals that mention the gold value) |
| govrag (strict, dense) | 38.2 | 35.0 | 71.7 |
| hybrid (BM25+dense RAG control) | 41.6 | 26.2 | 77.2 |
| baseline (vanilla dense RAG) | 41.4 | 26.0 | see metrics json |
| closedbook | see metrics json | – | see metrics json |
| llamaindex_k2, llamaindex_k8 | see metrics json | – | see metrics json |
| rankify | any_gold 92.0 (unjudged) | – | 78.1 (unjudged) |
| openwebui | 462/500 items only (checkpoint) | – | not run |

GLM-5.3-Flash three-way verdicts exist for baseline, govrag, closedbook, hybrid, llamaindex_k2,
llamaindex_k8 on both datasets: `results_external/judge3_table_z-ai-glm-5-3-flash.md`.
Reference points from it: CONFLICTS GOV-RAG 69.4 FC / 21.2 Inc vs hybrid RAG 68.6 / 17.2;
RAMDocs GOV-RAG 46.8 / 9.6 vs hybrid RAG 51.8 / 11.0.

Files per run in `results_external/`: `<tag>_metrics_<stamp>.json`, `<tag>_details_<stamp>.csv`
(judge input), `<tag>_<model>_checkpoint.jsonl` (resume state). Console logs are not committed.

## Setup on the H200

```bash
git clone https://github.com/fridayyy3000/ConflictingBench.git && cd ConflictingBench
git checkout eval/baseline-comparison
python -m venv .venv && source .venv/bin/activate
pip install numpy==1.26.4 openai sentence-transformers rank-bm25 pypdf python-docx requests
```

Copy from the Mac (both are gitignored on purpose; do not re-download the datasets, the exact
local copies are what the result files were produced from):

```bash
# run on the Mac; replace user@h200 and the clone path
rsync -av ~/Documents/Conflicting_documents/data/external/   user@h200:ConflictingBench/data/external/
rsync -av ~/Documents/Conflicting_documents/data/embeddings/  user@h200:ConflictingBench/data/embeddings/
```

`data/external/` is 47 MB (RAMDocs + CONFLICTS raw files). `data/embeddings/BAAI-bge-small-en-v1-5.sqlite`
is the 24 MB embedding cache; only needed if any generation is rerun.

Secrets: the NVIDIA key lives in `corvic/.env` on the Mac (gitignored). Recreate it by hand on
the H200 only if you regenerate answers. For judging against a local vLLM the judge still
requires a key variable to be set; any value works: `export OPENAI_API_KEY=EMPTY`.

## Judge: start the model

Pick one open model and do not benchmark judges. Keep the judge outside the generator's
family (Nemotron). Two that fit one H200:

```bash
pip install vllm
vllm serve Qwen/Qwen3-32B --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90
```

or `openai/gpt-oss-120b` (fits in its released MXFP4 form). The judge strips `<think>…</think>`
blocks and asks for JSON, so reasoning models are fine; `--max-tokens 4096` in the judge covers
reasoning plus the verdict.

## Judge: run every row

Smoke first (20 rows), then everything. The judge resumes from its checkpoint, so it is safe to
rerun the same command after an interruption.

```bash
cd corvic && export OPENAI_API_KEY=EMPTY
J="--judge-model Qwen/Qwen3-32B --base-url http://localhost:8000/v1 --workers 16"
python judge_three_way.py ../results_external/ramdocs_nvidia_govrag_hybrid_answerandflag_details_*.csv $J --limit 20
```

Judge the newest details CSV of every run tag:

```bash
cd corvic && export OPENAI_API_KEY=EMPTY
J="--judge-model Qwen/Qwen3-32B --base-url http://localhost:8000/v1 --workers 16"
for tag in $(ls ../results_external/*_details_*.csv | sed -E 's|.*/||; s/_details_[0-9_]+\.csv$//' | sort -u); do
  f=$(ls -t ../results_external/${tag}_details_*.csv | head -1)
  echo "== $tag  ($f)"; python judge_three_way.py "$f" $J || echo "FAILED $tag"
done
```

Build the table (slug = judge model with non-alphanumerics replaced by `-`):

```bash
cd corvic && python build_judge_table.py --judge-slug Qwen-Qwen3-32B
# writes ../results_external/judge3_table_Qwen-Qwen3-32B.md
```

Load estimate: ~19 rows × ~480 items ≈ 9,000 verdicts, plus the 3 CONFLICTS variant rows
once they land; a 32B judge on one H200 at 16 workers does this in a few hours.

Judge sanity check (do this before trusting the table): the summary json of each run has
`n_labelled` and FC/PC/Inc. Compare the new judge to GLM on the six rows both judged
(`*_judge3_z-ai-glm-5-3-flash_*.json`): FC and Inc should move together across methods and
rankings should match; a large disagreement on one method means the new judge reads
abstentions or hedges differently and needs a prompt look before the table is used.

## CONFLICTS variant runs: done (24 Nov 2026, 14:30)

All three variant rows finished on the Mac with no retry warnings and are committed with
their metrics, details CSVs and checkpoints. The self-contained H200 folder (`h200/`,
assembled by `h200/make_bundle.sh` into `h200_bundle/`, copied to
`/mnt/data1/gnajpand/conflictbench_h200/`) carries the newest details CSV of every row; re-run
the bundle script and the rsync after any new run.

## Parked (not needed for the table; the user's call)

- Open WebUI: 38 remaining RAMDocs items + all 458 CONFLICTS items (needs the Open WebUI
  container, local MiniLM embedding; CPU heavy). If not completed, drop the row.
- Vanilla RAG RAMDocs: 3 items are transport fallbacks (user said skip).
- Harness `--base-url` for a self-hosted *generator*: `corvic/pipeline/providers.py` still
  defaults to the NVIDIA endpoint; the `paper_pipeline` copy reads `DOCUINSIGHT_BASE_URL`.
  Only matters if answers are regenerated on the H200.
- Governance-bearing evaluation (fictional pack) if a reviewer asks; one day on the H200.

## After judging: package and demo (not H200 work)

Order agreed 23 Nov: freeze the configuration (done, `paper_pipeline/`) → pip package from it
(pyproject, entry point `docuinsight`, example corpus bundled, `serve` command) → UI: reader
pane with sentence highlighting (text view default, open-the-original PDF/Word view on
demand, side-by-side conflicts, decision badge, corroboration strip) → folder upload with
per-document governance panel → URL/S3 ingest. Name for paper and package still undecided.
