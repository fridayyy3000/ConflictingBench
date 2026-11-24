# Comparable systems for the GOV-RAG demo paper (scan of 2026-11-20)

Two scans: (A) all ACL-family System Demonstrations volumes 2024-2026 (ACL 2024, EMNLP 2024, NAACL 2025, ACL 2025, EMNLP 2025, ACL 2026, EMNLP 2026, AACL 2026; 457 papers; no NAACL 2026 volume exists), (B) deployed open-source RAG products and libraries, checked against GitHub and official docs.

Requirements for a usable baseline: public code, accepts arbitrary per-question documents, configurable OpenAI-compatible LLM endpoint (NVIDIA integrate.api.nvidia.com, Nemotron), scriptable over ~1000 questions.

## Finding 1: no demo-track system does conflict-aware QA over user documents
Only one demo-track system has any knowledge-conflict handling: KMatrix-2 (EMNLP 2025 demos, 2025.emnlp-demos.42), a GUI toolkit that exposes inter-context and context-memory conflict-resolution strategies (ICL, Discriminator, COIECD, CAD) and returns an answer, not a structured decision with citations; academic-use-only license; OpenAI-compatible endpoint unverified; no CLI confirmed. Adjacent but not comparable (web-search evidence, or no code): VeraCT Scan (ACL 2024), Medico (EMNLP 2024), FACTS&EVIDENCE (NAACL 2025), FactSearch (ACL 2026), SCOPE (EMNLP 2026), PRAISE (ACL 2025), ReqLens (AACL 2026). SentryLine (EMNLP 2026 demos, 2026.emnlp-demos.85) is evidence-grounded QA over evolving documents, our own prior work.

## Finding 2: demo-track RAG toolkits that can run our documents
| System | Venue | LLM endpoint | Per-question docs | Batch | Status |
|---|---|---|---|---|---|
| Rankify | ACL 2026 demo (2026.acl-demo.21) | OpenAI / LiteLLM backends (base_url unverified) | Yes, passages passed as Context objects | Yes | Apache-2.0, active (Sep 2026) |
| FlexRAG | ACL 2025 demo (2025.acl-demo.60) | OpenAIGeneratorConfig.base_url verified | Corpus prep per question = extra plumbing | Partial | MIT, active (Aug 2026) |
| FlashRAG | WWW 2025 resource (not demo track) | base_url verified | JSONL corpus + index | Yes, pipeline.run | MIT, active (Oct 2026) |
| RAGLAB | EMNLP 2024 demo | HF/vLLM only | Wikipedia-centric | Yes | MIT, stale since Oct 2024 |
Not runnable: TruthReader (fixed fine-tuned models), LocalRQA (stale), RAGViz (fixed corpora), InspectorRAGet (eval UI, archived), Ai2 Scholar QA (Semantic Scholar corpus).

## Finding 3: deployed products and libraries
| System | Stars | License | OpenAI-compatible LLM | bge-small in-process | Scriptable ingest->query | Per-question isolation | Defaults |
|---|---|---|---|---|---|---|---|
| Open WebUI | 159k | own BSD-derived | env OPENAI_API_BASE_URL | yes (RAG_EMBEDDING_MODEL) | REST files + chat/completions with file ids | yes, file list per call | chunk 1024/100, top-k 10, fixed RAG_TEMPLATE |
| AnythingLLM | 68.5k | MIT | env GENERIC_OPEN_AI_* | no (ONNX MiniLM or embedding server) | REST workspace/upload/chat | workspace per question | chunk 1000/20, topN 4, thr 0.25 |
| RAGFlow | 94.5k | Apache-2.0 | native NVIDIA + OpenAI-compatible factories | no (external embedding endpoint) | ragflow-sdk dataset/upload/parse/chat | dataset per question (async parse) | chunk 256 tok, top_n 6, max_token 512, reranker |
| LlamaIndex | 53k | MIT | OpenAILike(api_base) | yes (HuggingFaceEmbedding) | Python, fresh VectorStoreIndex per question | yes | chunk 1024/20, similarity_top_k 2 |
| Haystack | 27k | Apache-2.0 | OpenAIChatGenerator(api_base_url) | yes | Python pipeline, InMemoryDocumentStore | yes | split 200 words, top_k 10 |
| LangChain | 150k | MIT | ChatOpenAI(base_url) (docs disclaim 3rd-party) | yes | Python, no canonical default pipeline any more | yes | tutorial 1000/200, k 4 |
| LightRAG | 40.8k | MIT | openai_complete_if_cache(base_url) | yes (EmbeddingFunc) | Python insert/query | working_dir per question | graph build per question; mode naive possible |
Excluded: Verba (archived), Quivr (pivoted), R2R (stale), Kotaemon (no API, slowing), GraphRAG (maintenance mode, per-question graphs), Dify (app bound to fixed datasets), Onyx (Vespa + model servers, heavy), PrivateGPT/khoj (lower fit), NotebookLM (Gemini only, no query API; manual qualitative reference at most). None has conflict or credibility handling.

## Recommendation for the main table
1. Direct CoT (no retrieval), vanilla RAG, hybrid RAG: our harness.
2. Library default: LlamaIndex default query engine (OpenAILike + HuggingFaceEmbedding bge-small). Report pure default (top_k 2) and top_k 8 matched to GOV-RAG.
3. Deployed product: Open WebUI document chat (same bge-small embedder, LLM via one env var, per-question file list). Caveats: fixed RAG_TEMPLATE, rejects duplicate content (RAMDocs duplicates need a suffix), async processing.
4. Demo-track toolkit: Rankify (ACL 2026 demo) standard RAG strategy, passages passed directly. Confirms the comparison includes a peer demo-track system.
5. Conflict-aware peer, sample only: KMatrix-2 (EMNLP 2025 demo) on 50-100 questions if its generator can be pointed at Nemotron; otherwise cite as the only conflict-aware demo and explain why it is not comparable.
Optional: NotebookLM hand-run sample as the production reference (analogue of ASCO assistant in SentryLine).
Appendix: GOV-RAG ablations (prompt-only, no validator, no diversification); reported MADAM-RAG / Astute RAG set-EM on RAMDocs quoted with their model.
