# paper_pipeline — the system configuration used in the paper

This folder holds the pipeline **as it will be reported in the paper** ("Variant 3"),
separated from the evaluation harness, baselines and result files in `corvic/`.
It is the seed of the pip package: a plain Python package `docuinsight/` with no
server, no benchmark code and no Google Cloud dependency.

## The configuration

| Stage | Setting | Where |
|---|---|---|
| First-stage retrieval | BM25 + dense, reciprocal-rank fusion, top 30 chunks | `core.hybrid_retrieve` |
| Reranking | 0.3 semantic + gate × (0.5 authority + 0.2 scope) | `core.build_candidates` |
| Conflict diversification | one chunk per numeric claim value, 8 chunks kept | `core.diversify_conflicts` |
| Resolution policy | **answer-and-flag**: governance → corroboration (distinct documents) → consistency → tie | `core.SYSTEM_PROMPT_ANSWER_AND_FLAG` |
| Validation | allowed decision; every citation a verbatim quote from a retrieved chunk; one repair, then salvage | `core.run_generation` |
| Embedder | `BAAI/bge-small-en-v1.5`, local | `providers.LocalEmbedder` |
| Generator | any OpenAI-compatible chat endpoint (paper: `nvidia/nemotron-3.5-super-vl-120b-a12b`) | `providers.OpenAICompatGenerator` |

Defaults in code: `GovRAGGemini.query(retrieval="hybrid", policy="answer_and_flag")` and
`OpenAICompatGenerator.generate(policy="answer_and_flag")`.

### Ablations (same code, arguments only)

| Paper row | `--retrieval` | `--policy` | Harness run tag in `results_external/` |
|---|---|---|---|
| **System (Variant 3)** | hybrid | answer_and_flag | `*_nvidia_govrag_hybrid_answerandflag` |
| policy only (Variant 2) | dense | answer_and_flag | `*_nvidia_govrag_dense_answerandflag` |
| retrieval only (Variant 1) | hybrid | strict | `*_nvidia_govrag_hybrid_strict` |
| refuse-on-conflict (original GOV-RAG) | dense | strict | `*_nvidia_govrag` |

RAMDocs (500 items, set exact match, Nemotron 3.5 Super): 45.6 / 43.6 / 37.6 / 38.2 in the
order above; hybrid RAG control 41.6, vanilla RAG 41.4. Decided 23 Nov 2026.

## Usage

```bash
pip install -r requirements.txt
export NVIDIA_API_KEY=...          # or OPENAI_API_KEY; or pass --env-file path/to/.env
python -m docuinsight ask --docs ./my_documents "What is the travel reimbursement limit?"
```

Options: `--model`, `--base-url` (any OpenAI-compatible endpoint, e.g. a local vLLM),
`--embed-model`, `--retrieval dense`, `--policy strict` (ablations), `--json`.
Environment: `NVIDIA_API_KEY`/`OPENAI_API_KEY`, `NEMOTRON_MODEL`, `DOCUINSIGHT_BASE_URL`,
`EMBED_DEVICE` (`cpu`, `mps`, `cuda`).

From Python:

```python
from docuinsight import load_folder, build_pipeline
pipe = build_pipeline(load_folder("./my_documents"))
result = pipe.query("What is the travel reimbursement limit?")
result["decision"], result["answer"], result["citations"]
```

## Relation to `corvic/`

`docuinsight/core.py` and `providers.py` are copies of `corvic/pipeline/gov_rag_gemini.py`
and `providers.py` with three changes: package-relative imports, Google SDKs optional
(no hard exit when absent), and the Variant 3 defaults above. `chunker.py` and
`document_parser.py` are unchanged copies. The evaluation harness keeps using
`corvic/pipeline/` so result files remain reproducible; port fixes both ways until the
harness is switched to import this package.

Not in this folder on purpose: the FastAPI server and web UI (`corvic/main.py`,
`corvic/static/`), the baselines, the judge, datasets and results.
