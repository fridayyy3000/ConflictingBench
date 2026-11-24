# Conflict-aware RAG / QA METHODS (2023 – Nov 2026): catalogue, baselines, protocol, gains, code, open problems

Scope note: this file covers METHODS (not datasets) that explicitly handle conflicting or unreliable retrieved evidence. Dataset papers are mentioned only where needed for evaluation protocol or headline numbers. All numbers below are copied from the primary paper (arXiv HTML/abs or ACL Anthology) unless flagged. Items dated 2026 were checked against arXiv/ACL pages on 2026-11-14; several carry internal inconsistencies that are flagged in place. Research was conducted with ~60 search/fetch calls; where a full-text fetch failed (PDF-only), that is recorded as a gap.

---

## KQ1. What do the named and newer conflict-aware methods do, and how well? (method catalogue)

### Takeaway
The field splits into six mechanism families: (a) multi-agent debate (MADAM-RAG), (b) internal/external knowledge consolidation (Astute RAG, CARE-RAG), (c) credibility-conditioned generation via fine-tuning (CAG, QACC Explain-and-Answer, reasoning-trace SFT), (d) training-free contrastive decoding (CAD, AdaCAD, COIECD, CoCoA), (e) detect-then-resolve pipelines with light classifiers / probes / soft prompts (Discern-and-Answer, ConflictRAG, TCR, ProbeRAG, CARE soft-prompting, FaithfulRAG), and (f) source-reliability / trust / temporal weighting (RA-RAG, TrustPropRAG, Re3, FRESCO) plus belief-based abstention (ERA, SABER, EvidentialRAG). The strongest *peer-reviewed* inter-document-conflict result on a dedicated conflict benchmark is MADAM-RAG on RAMDocs (34.40 EM, Llama-3.3-70B-Instruct; COLM 2025), and it is still far from solved.

### Cited Findings

#### A. MADAM-RAG (multi-agent debate) — Wang, Prasad, Stengel-Eskin, Bansal; COLM 2025; arXiv 2504.13079
- Paper: "Retrieval-Augmented Generation with Conflicting Evidence", v1 17 Apr 2025, v2 12 Aug 2025, COLM 2025 — [arXiv abs](https://arxiv.org/abs/2504.13079v2)
- Mechanism: each retrieved document is assigned its own LLM agent; agents "debate over the merits of an answer over multiple rounds"; an aggregator "collates responses for disambiguated entities while discarding misinformation and noise". Max rounds T = 3 with early stop when all agents keep their previous answers; Appendix D says convergence "within 1-2 rounds for most examples" — [arXiv HTML v2](https://arxiv.org/html/2504.13079v2)
- Metric: strict exact match — output correct "if and only if it includes all gold answers and no incorrect or misleading answers supported only by misinformation documents"; Precision/Recall/F1 also reported in §6.1 — [arXiv HTML v2](https://arxiv.org/html/2504.13079v2)
- Main results (Table 1, EM %), per model / method → FaithEval | AmbigDocs | RAMDocs — [arXiv HTML v2](https://arxiv.org/html/2504.13079v2):
  - SelfRAG-Llama2-13B (Self-RAG): 12.40 | 55.00 | 26.80
  - Llama3.3-70B-Inst — No RAG 26.70 | 4.30 | 5.80; Prompt-based (concat) 27.30 | 54.20 | 32.60; Single Agent w/ Self-reflection 25.70 | 54.50 | 30.40; Speculative RAG 41.80 | 44.30 | 30.60; Astute RAG 37.10 | 46.80 | 31.80; **MADAM-RAG 43.10 | 58.20 | 34.40**
  - Qwen2.5-72B-Inst — No RAG 26.40 | 1.80 | 4.20; Prompt-based 38.50 | 41.20 | 20.60; Self-reflection 39.40 | 29.40 | 21.80; Speculative RAG 56.20 | 13.40 | 22.20; Astute RAG 44.60 | 39.80 | 20.80; **MADAM-RAG 57.70 | 52.70 | 26.40**
  - GPT-4o-mini — No RAG 31.00 | 1.00 | 2.50; Prompt-based 21.00 | 51.50 | 25.00; Self-reflection 36.00 | 44.50 | 22.50; Speculative RAG 36.50 | 22.50 | 23.00; Astute RAG 34.00 | 15.00 | 13.00; **MADAM-RAG 38.50 | 63.00 | 28.00**
- Abstract-level claims: "up to 11.40%" over strong RAG baselines on AmbigDocs and "up to 15.80%" absolute on FaithEval with Llama3.3-70B-Instruct; RAMDocs best baseline 32.60 EM — [arXiv abs](https://arxiv.org/abs/2504.13079v2)
- Ablation (Llama3.3-70B, Fig. 3 text): RAMDocs round-1 accuracy 30.00 without aggregator vs 37.80 with; F1 59.79 vs 68.63; with aggregator, RAMDocs precision 67.41 / 68.52 / 68.57 over rounds 1-3 — [arXiv HTML v2](https://arxiv.org/html/2504.13079v2)
- Robustness: as the imbalance between supporting docs and misinformation grows, concatenated-prompt baseline falls "by up to 8%", MADAM-RAG average variance 3.33 across rounds; as misinformation increases, concatenated-prompt and Astute RAG each "suffer a decrease of 46%" while MADAM-RAG is highest at each level — [arXiv HTML v2](https://arxiv.org/html/2504.13079v2)
- Code/data: https://github.com/HanNight/RAMDocs — MIT license; contains `RAMDocs_test.jsonl`, `run_madam_rag.py`, `requirements.txt`; args `model_name`, `cache_dir`, `data_path`, `num_rounds`, `seed`; example uses `meta-llama/Llama-3.3-70B-Instruct` via HF; README has no baseline implementations and no results table — [GitHub](https://github.com/HanNight/RAMDocs)
- Independent re-implementation (2026, poisoning setting, GPT-5-mini, NQ): MADAM-RAG clean accuracy only 56.6% vs 92.0% vanilla RAG, with a 41.4% non-answer rate (29.6% hedging, 11.7% unknown); CorruptRAG-AK attack success 51.9% vs 83.0% vanilla / 45.9% agentic RAG / 27.0% RLM; MADAM-RAG consumed 259 of 359 total compute hours (72%); authors attribute weakness partly to "implementation divergences" (API model, sequential agents, raw peer responses) — Korn, arXiv 2605.05632, [arXiv HTML](https://arxiv.org/html/2605.05632); code https://github.com/samkorn/rag-poisoning-architecture-bench

#### B. Astute RAG — Wang, Wan, Sun, Chen, Arık (Google); ACL 2025; arXiv 2410.07176
- v1 9 Oct 2024, v2 May 2025, "accepted to ACL 2025" — [arXiv abs](https://arxiv.org/abs/2410.07176v1); [Google Research page](https://research.google/pubs/astute-rag-overcoming-imperfect-retrieval-augmentation-and-knowledge-conflicts-for-large-language-models/)
- Mechanism (3 steps): (1) adaptive internal-knowledge generation — LLM writes at most m̂ passages (default ≤1) from memory, may return none; (2) iterative source-aware consolidation of internal + external passages over t iterations; (3) answer finalization — propose an answer per consolidated passage group, select the most reliable. Default t=1 → 2 API calls/query; t=2 → 3 calls; t=3 → 4 calls — [arXiv HTML v1](https://arxiv.org/html/2410.07176v1)
- Setup: Google Search, top-30 → first 10 accessible pages (10 passages/question); 1,042 short-form QA pairs across NQ, TriviaQA, BioASQ, PopQA; ~70% of retrieved passages do not contain the answer; ~20% of questions have no passage with the answer (NQ 34%, TriviaQA 18%, BioASQ 24%, PopQA 50%) — [arXiv HTML v1](https://arxiv.org/html/2410.07176v1)
- Claude 3.5 Sonnet overall accuracy (calls): No RAG 54.51 (1); RAG 55.47 (1); USC 58.73 (4); GenRead 53.55 (2); RobustRAG 56.53 (11); InstructRAG 58.83 (1); Self-Route 58.06 (1-2); **Astute t=1 61.71 (2); t=2 62.67 (3); t=3 62.86 (4)**. Per-dataset for Astute t=1: NQ 52.20, TriviaQA 84.10, BioASQ 60.14, PopQA 44.38 — [arXiv HTML v1](https://arxiv.org/html/2410.07176v1)
- Gemini 1.5 Pro overall: No RAG 51.34; RAG 53.65; USC 56.43; GenRead 54.70; RobustRAG 45.59 (high refusal rate); InstructRAG 56.14; Self-Route 57.58; **Astute t=1 59.21; t=2 59.69; t=3 59.21** — [arXiv HTML v1](https://arxiv.org/html/2410.07176v1)
- Conflict analysis (Claude): 19.2% of items have internal-vs-external answer conflict with one correct; internal correct 47.4% / external correct 52.6% of those; Astute picks the correct one ~80% of the time. Worst case (RGB, all passages negative): RAG vs No RAG gap >50 points; Astute "the only method" close to No RAG (exact values only in Fig. 4c image) — [arXiv HTML v1](https://arxiv.org/html/2410.07176v1)
- Code: no official repository found on the arXiv page, the Google Research page, or via search — [search](https://www.alphaxiv.org/resources/2410.07176) lists only an "Implementation" tab with no URL in snippet. UNVERIFIED whether any official release exists; MADAM-RAG and TCR papers re-implemented it as a baseline.
- Limitations stated: depends on strong instruction-following LLMs; short-form only — [arXiv HTML v1](https://arxiv.org/html/2410.07176v1)

#### C. Credibility-aware Generation (CAG) — Pan et al.; EMNLP 2024 main; arXiv 2404.06809
- "Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation", EMNLP 2024 pp. 19844–19863 — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1109); [arXiv abs](https://arxiv.org/abs/2404.06809)
- Mechanism: each retrieved document is prefixed with a credibility label (high/medium/low); the LLM is instruction-fine-tuned on data where GPT-3.5 writes credibility-guided explanations. Credibility = min(adjusted relevance level, source level). Relevance: SPLADE retriever score binned into 3 levels (equal count or equal interval). Timeliness: gap between query time and publication date; if gap exceeds a validity period (obtained by prompting an LLM 3× and voting), relevance level is downgraded. Source reliability: set per scenario (LLM-generated fake news = low; reputable news sites = high) — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3)
- Models fine-tuned: LLaMA-2-7B (CAG-7B), LLaMA-2-13B (CAG-13B), Mistral-7B (CAG-mistral-7B); "vanilla IFT" ablation on LLaMA-2-7B without credibility — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3)
- Benchmark: open-domain QA (HotpotQA 500, 2WikiMHQA 500, MuSiQue 500, ASQA 948, RGB 300), time-sensitive QA (EvolvingTempQA 321), misinformation-polluted QA (NewsPollutedQA 480) — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3)
- Table 2 (EM; RGB/EvolvingTempQA/NewsPollutedQA at noise 0.8/0.8/0.75), order HotpotQA | 2Wiki | MuSiQue | ASQA | RGB | EvolvingTempQA | NewsPollutedQA — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3):
  - Retrieval-based ChatGPT 0.334 | 0.368 | 0.194 | 0.404 | 0.773 | 0.579 | 0.231; LLaMA-2-70B 0.418 | 0.390 | 0.256 | 0.316 | 0.823 | 0.526 | 0.430
  - Retrieval+reranking ChatGPT 0.396 | 0.394 | 0.216 | 0.388 | 0.790 | 0.632 | 0.427; LLaMA-2-70B 0.422 | 0.504 | 0.320 | 0.388 | 0.833 | 0.570 | 0.306
  - Retrieval+credibility (labels given in prompt, no training) ChatGPT 0.422 | 0.402 | 0.182 | 0.440 | 0.807 | 0.673 | 0.408; LLaMA-2-70B 0.398 | 0.402 | 0.262 | 0.492 | 0.817 | 0.536 | 0.279
  - **CAG-7B 0.509 | 0.578 | 0.340 | 0.496 | 0.897 | 0.826 | 0.442; CAG-13B 0.514 | 0.604 | 0.408 | 0.525 | 0.917 | 0.829 | 0.483; CAG-mistral-7B 0.502 | 0.540 | 0.384 | 0.505 | 0.900 | 0.835 | 0.613**
- Noise robustness (RGB noise 0.2/0.4/0.6/0.8): ChatGPT 0.917/0.913/0.850/0.773; LLaMA-2-70B 0.960/0.937/0.910/0.823; CAG-13B 0.977/0.967/0.943/0.917. NewsPollutedQA (0.5/0.67/0.75): ChatGPT 0.340/0.250/0.231; CAG-mistral-7B 0.679/0.640/0.613 — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3)
- Cost: single generation per query at inference (LLM calls not stated; 3-shot for baselines). Code, benchmark, models: https://github.com/panruotong/CAG (license not stated on paper page) — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3); [GitHub](https://github.com/panruotong/CAG)

#### D. Context-aware Decoding (CAD) and descendants (AdaCAD, COIECD, CoCoA)
- CAD — Shi, Han, Lewis, Tsvetkov, Zettlemoyer, Yih; arXiv 2305.14739 (24 May 2023); NAACL 2024 short per ACL Anthology (one review site says NAACL 2023 — venue year conflict flagged) — [arXiv abs](https://arxiv.org/abs/2305.14739v1); [ACL Anthology](https://preview.aclanthology.org/setup/2024.naacl-short.69)
  - Formula: p̃(y_t | c,x,y<t) ∝ softmax[(1+α)·logit(y_t | c,x,y<t) − α·logit(y_t | x,y<t)], α = 1 for knowledge-conflict QA; training-free, 2 forward passes per token — [arXiv HTML v1](https://arxiv.org/html/2305.14739v1)
  - Table 3 EM (Regular → CAD) MemoTrap | NQ | NQ-SWAP: OPT-13B 32.5→44.5 | 29.2→32.2 | 18.8→36.9; OPT-30B 28.4→41.0 | 29.4→35.5 | 14.7→29.0; GPT-Neo-20B 37.1→57.3 | 22.8→32.1 | 16.1→36.8; LLaMA-13B 23.8→57.1 | 22.3→33.6 | 11.7→36.7; LLaMA-30B 25.8→50.6 | 23.8→34.0 | 9.6→37.7; FLAN-T5-XXL 82.0→88.7 | 85.5→82.5 | 73.0→77.1 — [arXiv HTML v1](https://arxiv.org/html/2305.14739v1)
  - Code: https://github.com/xhan77/context-aware-decoding — [search result quoting paper](https://arxiv.org/abs/2305.14739v1)
- AdaCAD — Wang, Prasad, Stengel-Eskin, Bansal; arXiv 2409.07394 (11 Sep 2024); NAACL 2025 long pp. 11636–11652. Replaces fixed α with per-token Jensen-Shannon divergence between contextual and parametric distributions; training-free; reports +14.21% absolute average QA accuracy over a static contrastive baseline across 4 LLMs and 6 QA datasets, and +6.19 AlignScore on 3 summarization sets (v1 says 4.16 over greedy — version discrepancy flagged). Code https://github.com/HanNight/AdaCAD — [arXiv abs v2](https://arxiv.org/abs/2409.07394v2); [ACL Anthology](https://preview.aclanthology.org/setup/2025.naacl-long.581)
- CoCoA — Khandelwal, Gupta, Agrawal (Microsoft India); arXiv 2508.17670 (25 Aug 2025); EMNLP 2025 main. Blends p_ctx^λt · p_prior^(1−λt) with λ_t from Rényi divergence (α=0.5), entropy gap, and contextual peakedness (z=5.0, γ=1.0). Table 1 EM (Greedy / CAD / COIECD / AdaCAD / CoCoA), NQ-SWAP: Llama2-13B 54.89 / 80.35 / 59.84 / 67.84 / 80.36; Llama3-8B 47.81 / 77.94 / 51.16 / 62.34 / 79.15; Llama3-70B 55.74 / 81.32 / 57.26 / 70.07 / 88.32; Mistral-7B 56.86 / 66.89 / 58.09 / 67.27 / 80.90. Average over 6 QA sets, Llama3-70B: Greedy 66.50, CAD 55.77, COIECD 65.16, AdaCAD 68.68, CoCoA 77.87. Note: CAD badly hurts non-conflict sets (Llama3-8B TriviaQA 85.70→40.20). Some "Avg" cells do not equal the row mean (flagged by fetch). Code https://github.com/infusion-zero-edit/CoCoA/ — [arXiv HTML v1](https://arxiv.org/html/2508.17670v1); [ACL Anthology](https://preview.aclanthology.org/setup/2025.emnlp-main.348)

#### E. Discern-and-Answer — Hong, Kim, Kang, Myaeng, Whang; NAACL 2024 Findings; arXiv 2305.01579
- "Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise", v1 2 May 2023, v3 9 Jun 2024 — [arXiv abs](https://arxiv.org/abs/2305.01579)
- Variants: Disc^FiD — one-layer discriminator jointly trained with FiD (T5-base) to label each passage perturbed/not (QA loss + BCE + contrastive); Disc^Inst — GPT-3.5 (text-davinci-003) prompted to identify perturbed docs before answering; hybrid — Disc^FiD decisions inserted into the GPT-3.5 prompt — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- MacNoise: NQ-open and TriviaQA-open; GPT-3.5-turbo (train) / GPT-4 (eval) rewrite passages so the answer is removed/negated/entity-replaced; eval perturbation ratios 0/15/25/35% — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- MacNoise dev EM, NQ (0/15/25/35%, avg): FiD semi-parametric 62.5/50.8/39.1/28.5 (45.2) → +Disc^FiD 62.5/52.0/41.4/30.1 (46.5); GPT-3.5 semi-parametric 50.4/28.5/23.8/16.0 (29.7) → +Disc^Inst 48.8/36.3/28.5/19.5 (33.3) → +Disc^FiD 51.2/37.1/30.1/21.5 (35.0). TriviaQA: GPT-3.5 57.3 avg → Disc^Inst 59.9 → Disc^FiD 64.7 — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Entity-perturbation NQ test EM avg: GPT-3.5 40.1 → Disc^Inst 41.3 → Disc^FiD 43.9 — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Code: https://github.com/wjdghks950/Discern-and-Answer; limits: 256 dev instances only (cost), top-5 docs for GPT-3.5 — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Independent replication on real conflicts (QACC, GPT-4o few-shot): Context 51.17 EM-total vs Discern&Answer 48.83 vs Majority 50.55 — Discern-and-Answer *hurt* GPT-4o and Claude-3 (48.71 → 40.71) on naturally conflicting web contexts — Liu et al., [arXiv HTML 2410.12311v3](https://arxiv.org/html/2410.12311v3)

#### F. QACC "Explain-and-Answer" + explanation fine-tuning — Liu, Roth et al.; Findings NAACL 2025; arXiv 2410.12311
- Human-annotated QACC (AmbigQA unambiguous Qs, top-10 Google results; conflict rate ~25%: train 394 (29%), dev 303 (19%), test 813 (25%)); 3 LLMs evaluated (GPT-4o, Claude-3-Sonnet, Phi-3-Medium 14B) — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3); [ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-naacl.99/)
- Method: LoRA SFT of Phi-3-Medium on human one-sentence explanations of why an answer was chosen. Table 2 EM-total: Phi-3 few-shot Context 49.57; +Majority 52.52; +Discern&Ans 52.03; +Exp&Ans 52.64; Phi-3 FT Context 49.57; FT+Majority 53.01; FT+Dis&Ans 46.25; **FT+Exp&Ans 54.74** (EM on conflicting subset 47.34 vs 42.51 few-shot). Transfers to perturbed NQ-Open (0/25/50%): FT Context 34.24/32.27/29.67 → FT+Exp&Ans 37.31/35.76/33.18 — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3)
- Flagged: text reports 2.9 EM gain but table implies ~5.2; setup says GPT-4o-mini but tables say GPT-4o. Code/data "will be released"; search result says released with the NAACL version — [papers.cool listing](https://papers.cool/venue/2025.findings-naacl.99@ACL) (UNVERIFIED URL)

#### G. Detect-then-resolve pipelines and probes (2025–2026)
- **ConflictRAG** — Wang, Li, Liu, Shu; arXiv 2605.17301 (v1 17 May 2026, v2 8 Jun 2026); "submitted to IEEE SMC 2026" (not peer-reviewed) — [arXiv abs](https://arxiv.org/abs/2605.17301)
  - Pipeline: two-stage pairwise conflict detection (embedding MLP on all-MiniLM-L6-v2 handles ~73% of pairs; pairs with confidence <0.7 go to the LLM) → classify (factual / temporal / opinion inter-document; parametric–contextual via closed-book vs open-book answer comparison) → resolve (factual: Entropy-TOPSIS ranking over 5 LLM-extracted criteria authority 0.312, recency 0.245, relevance, specificity, consistency; temporal: rank by recency and note change over time; opinion: multi-perspective synthesis with attribution; parametric: defer to retrieval) → generate with conflict annotations and a confidence qualifier — [arXiv HTML v1](https://arxiv.org/html/2605.17301v1)
  - Datasets: ConflictQA, NQ-Conflict (self-built), AmbigQA. Baselines: Standard RAG, RAG+Rerank, Self-RAG, CRAG, NLI-Filter (DeBERTa-v3-base), CoT Detection. GPT-4o-mini generator, GPT-4o judge. Correctness % (ConflictQA | NQ-Conflict | AmbigQA): Standard RAG 49.2 | 53.4 | 46.8; RAG+Rerank 52.3 | 55.8 | 49.1; Self-RAG 47.8 | 52.1 | 45.2; CRAG 54.1 | 57.2 | 51.2; NLI-Filter 57.8 | 61.5 | 55.1; CoT Detection 63.1 | 65.3 | 60.5; **ConflictRAG 68.9±1.4 | 71.4±1.2 | 65.8±1.8**. Detection 90.8% acc / 88.7% F1 (drops to 83.4% on AmbigQA). Cost $0.026/query vs $0.068 LLM-only (−62%). Judge bias estimated ~2.5%, corrected gain +3.1–3.7. Code "will be released upon acceptance" — [arXiv HTML v1](https://arxiv.org/html/2605.17301v1)
- **CARE-RAG (Conflict-Aware and Reliable Evidence)** — Chen, Bi, ... Liu; arXiv 2507.01281 (2 Jul 2025), preprint — [arXiv abs](https://arxiv.org/abs/2507.01281)
  - Mechanism: parameter-aware evidence (compare the model's parametric records), context-aware refinement, conflict-driven summarization by a distilled LLaMA-3.2-3B detector (fine-tuned on DeepSeek annotations), "QA Repair" of outdated gold answers. Top-5 evidences. Baselines: No RAG, RAG, InstructRAG, GenRead, Self-RAG. EM, NQ* | TriviaQA* | HotpotQA* | ASQA* | WikiQA*: Llama-3.2-8B [sic] RAG 40.3 | 66.1 | 35.3 | 33.2 | 33.9; InstructRAG 59.7 | 73.9 | 48.5 | 45.9 | 36.9; **CARE-RAG 63.9 | 79.6 | 55.9 | 52.6 | 47.1**; GPT-4.1-Nano InstructRAG 58.5 | 72.5 | 53.5 | 48.1 | 40.4 → **CARE-RAG 66.2 | 81.6 | 56.7 | 53.0 | 47.6**; avg +3.8 EM over InstructRAG. LLM calls not reported (≥4 implied). No code URL in paper — [arXiv HTML](https://arxiv.org/html/2507.01281). Flagged: "LLaMA-3.2-8B" is not a real Llama 3.2 size; standard QA sets, not conflict benchmarks.
- **CARE (Conflict-Aware soft prompting)** — Choi, Park, Lee, Lee; EMNLP 2025; arXiv 2508.15253 — [arXiv abs](https://arxiv.org/abs/2508.15253)
  - Context assessor encodes retrieved context into K=16 memory tokens; trained with grounded + adversarial soft prompting to emit a guidance signal (trust context vs parametric). Top-1 ColBERTv2 passage only. Avg over NQ/TriviaQA/WebQA/TruthfulQA/FactKG: Mistral-7B RAG 0.436, CAD 0.407, AdaCAD 0.430, SKR-kNN 0.434, **CARE 0.458** (+5.01% rel.); LLaMA-3-8B RAG 0.440, CAD 0.375, AdaCAD 0.422, Adaptive-RAG 0.452, **CARE 0.467** (+6.13%). Training: 2M Wikipedia contexts pretraining (~25 h) + NQ fine-tune (~3 h) on 2×A100. Code https://github.com/eunseongc/CARE. Limits: single passage, single seed — [arXiv HTML v2](https://arxiv.org/html/2508.15253v2)
- **TCR (Transparent Conflict Resolution)** — Ye et al.; arXiv 2601.06842 (11 Jan 2026), preprint — [arXiv abs](https://arxiv.org/abs/2601.06842)
  - Dual contrastive encoders (semantic match vs factual consistency) + self-answerability estimate → three scalars fed by SNR-weighted soft prompt; +0.3% params. Conflict detection F1 (Llama-3-8B): Prompt 71.2, KAFT 73.5, IRCAN 79.1, RAAT 66.4, Parenting 77.5, **TCR 84.3**. NQ single-page accuracy: Astute RAG 51.2/56.6/49.7, InstructRAG 47.8/51.2/50.3, Self-Route 46.5/49.2/48.5, **TCR 54.8/56.8/53.7** (Llama-3-8B/13B/Qwen3-8B). FLAGGED: Table 2 shows TCR MCOR (lower-better) *higher* than Prompt in every cell (61.9 vs 19.5) while the abstract claims −29.3 pp; "Llama-3-13B" is not a released size; no code URL — [arXiv HTML](https://arxiv.org/html/2601.06842). Treat with caution.
- **ProbeRAG** — Gao, Zhang, Bi, ... Su; ACL 2026 Findings; arXiv 2510.12460 (v1 14 Oct 2025, v3 15 Apr 2026) — [arXiv abs](https://arxiv.org/abs/2510.12460)
  - Knowledge pruning → latent conflict probing (conflicting vs aligned states linearly separable) → conflict-aware attention-head modulation. Datasets ConFiQA (QA/MR/MC), FaithEval, SQuAD(KRE). Baselines: Opin(Instr), KRE, FaithfulRAG, CAD, COIECD, AdaCAD, Context-DPO, CANOE, ParamMute. LLaMA-3.1-8B-Inst FaithEval EM: Full-Ctx 53.1, CAD 42.7, AdaCAD 47.7, FaithfulRAG 49.1, Context-DPO 53.7, CANOE 56.3, ParamMute 56.2, **ProbeRAG 64.4**; ConFiQA-QA EM Full-Ctx 49.0 → **ProbeRAG 91.7**. Code https://github.com/LinfengGao/ProbeRAG — [arXiv HTML v2](https://arxiv.org/html/2510.12460v2)
- **FaithfulRAG** — Zhang et al.; ACL 2025; arXiv 2506.08938; fact-level conflict identification + self-thinking before answering; code https://github.com/DeepLearnXMU/Faithful-RAG; numbers appear only as a baseline in ProbeRAG (above) — [arXiv abs](https://arxiv.org/abs/2506.08938)

#### H. Source-reliability / trust-weighted RAG
- **RA-RAG (Reliability-Aware RAG)** — Hwang et al. (POSTECH); arXiv 2410.22954 (v1 30 Oct 2024, v5 14 Oct 2025); listed as EMNLP 2025 main by papers.cool (UNVERIFIED against proceedings) — [arXiv abs](https://arxiv.org/abs/2410.22954); [papers.cool](https://papers.cool/venue/2025.emnlp-main.1738@ACL)
  - Mechanism: a "source" is a partition of the corpus (account, politician, news site). Offline: iterate (i) weighted-majority-vote answer per fact-checking query, (ii) reliability = fraction of a source's non-IDK answers matching the WMV answer, (iii) rescale v = N·w − 1, until convergence. AlignScore <0.1 → "I don't know"; DeBERTa-large NLI clusters paraphrases. Inference: κ-RRSS keeps first κ=4 reliable sources with non-IDK answers, top-3 docs per source, WMV aggregation — [arXiv HTML v5](https://arxiv.org/html/2410.22954v5)
  - Benchmark: NQ/TriviaQA/HotpotQA(single-hop), 1,600 queries each (200 for estimation, 1,400 test); GPT-4o-mini generates 9 paraphrased correct docs + 9 wrong answers × 3 misinformation docs; sources 4–9 (up to 1,000 for efficiency). Table A4 NQ accuracy with 1…7 adversaries of 9 (Llama3-8B-Inst): MV 0.740→0.327; Vanilla RAG 0.612→0.294; RobustRAG 0.676→0.201; **RA-RAG 0.737→0.558**; Oracle WMV 0.745→0.558. TriviaQA 7 adversaries: MV 0.477, Vanilla RAG 0.544, RA-RAG 0.722 — [arXiv HTML v5](https://arxiv.org/html/2410.22954v5)
  - Cost (NQ, GPT-4o-mini): 10 sources → 6.79 API calls, 4,251 tokens, $0.00065/query, acc 0.727 (vs 10 calls/0.744 without κ-RRSS); 1,000 sources → 8.66 calls. Latency/task: Vanilla RAG 0.32 s, Self-RAG 7.44 s, RA-RAG 1.21–1.82 s; offline estimation 4.65 min (5 sources) to 14.81 h (1,000). Code https://github.com/ml-postech/RA-RAG — [arXiv HTML v5](https://arxiv.org/html/2410.22954v5)
- **TrustPropRAG** — Li & Chen; Findings of EMNLP 2026 pp. 14926–14940 (Oct 2026). Document-relation graph; multi-hop trust propagation anchored by a limited set of human reliability feedback; trust scores used for rescoring and trust-aware prompting; claims EM gains over baselines and robustness to sparse/noisy feedback. No numbers, datasets, or code on the Anthology page — [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.711/)

#### I. Temporal / recency-aware methods
- **Re³ (Relevance & Recency Retrieval)** — Cao et al.; ACL 2026 long pp. 25735–25760. Time-Aware Dual Relevance Encoder (temporal signals in embedding space) + Conflict-Aware Recency Filter (list-wise comparison, suppresses outdated fact versions); Re²Bench >1.3M instances; +9.7% average generation accuracy over strongest baselines, max +25.2% on dynamic tasks; no code URL on the page — [ACL Anthology](https://aclanthology.org/2026.acl-long.1180/)
- **FRESCO** — An et al. (Meta Superintelligence Labs / UCLA); arXiv 2604.14227 (14 Apr 2026). Benchmark pairs recency-seeking queries with historical Wikipedia revisions; finds re-rankers have "a strong bias toward older, semantically rich documents"; proposes Pareto-optimal instruction optimization for rerankers, up to +27% on evolving-knowledge tasks; no code URL — [arXiv abs](https://arxiv.org/abs/2604.14227)
- **T-GRAG** — arXiv 2508.01680 (Aug 2025): time-stamped GraphRAG with temporal sub-query decomposition; Time-LongQA benchmark from annual reports — [arXiv abs](https://arxiv.org/abs/2508.01680v1) (numbers not fetched)
- **When Facts Change** — Wallat, Nejdl, Sikdar; Findings ACL 2026 pp. 2154–2184; WIKIRECENTCHANGES benchmark; prompting models to consider mutability "increases references to temporal change but does not improve factual accuracy"; smaller models rarely detect the conflict, larger detect but do not act — [ACL Anthology](https://aclanthology.org/2026.findings-acl.103/)
- CAG (above) is the only method found that uses *publication date vs LLM-estimated validity period* as an explicit input signal — [arXiv HTML v3](https://arxiv.org/html/2404.06809v3)

#### J. Belief / uncertainty-based abstention under conflict
- **ERA** — Shin, Cha, Lee, Park; arXiv 2604.20854 (page says "submitted 24 Feb 2026" though ID implies April 2026 — flagged); "Under Review". Dirichlet belief masses for internal vs external knowledge + Dempster-Shafer discordance; QLoRA (r=8) 1 epoch on 5k examples/dataset. Baselines Logprob, ICL, Self-Consistency, P_true, DTA. Overall F1 (answer+abstention): Qwen3-8B-Base DTA 0.465 → **ERA 0.545**; Llama-3.1-8B DTA 0.524 → **ERA 0.604**. Code https://anonymous.4open.science/r/ERA-318B — [arXiv HTML](https://arxiv.org/html/2604.20854)
- **SABER (Trust or Abstain?)** — Zhu et al.; arXiv 2605.18792 (11 May 2026). No fine-tuning of the LLM; K=3 reasoning traces each on parametric and context side (6 traces + 1 self-prior pass + self-eval passes); two lightweight predictors → 4-cell decision (trust PK / trust CK / either / abstain). Datasets ConFiQA, ConflictQA-PopQA, ConflictBank, TriviaQA, NQ; 4 backbones; 10 baselines (Closed-Book, Vanilla RAG, Internal-Eval, Context-Eval, Implicit/Explicit-SCR, TPC, TACS-LR, CR-DPO, R-Tuning). SABER Acc/MFS Llama-3.1-8B: ConFiQA 58.4/82.5, ConflictQA 53.5/91.0, ConflictBank 80.2/92.5, NQ 52.1/69.2. Claims risk-coverage curve Pareto-dominates every prompt-based abstainer. Code https://github.com/xizhu1022/SABER — [arXiv HTML v1](https://arxiv.org/html/2605.18792v1) (baseline numbers not extracted)
- **EvidentialRAG (ERAG)** — Hossain, Shayoni, Mridha; arXiv 2607.10491 (11 Jul 2026). Claim extraction → Dirichlet evidence → conflict-preserving Dempster-Shafer fusion → route to direct answer / conflict-aware answer / abstain. CRAG ambiguous subset: hallucination 45.3% (Corrective RAG) → 34.8% ("human-calibrated estimate"); conflict resolution 35.2% → 51.2%; ECE 0.122. No code — [arXiv abs](https://arxiv.org/abs/2607.10491)
- **Sufficient-context abstention** — arXiv 2411.06037: selective generation guided by sufficient-context signal improves fraction of correct answers among answered by 2–10% (Gemini, GPT, Gemma) — [search snippet](https://arxiv.org/pdf/2411.06037.pdf) (not fetched in full)

#### K. Reasoning-trace SFT with refusal
- **From Facts to Conclusions** — Mishra et al.; arXiv 2512.16795 (18 Dec 2025), "Under Review". Document-level adjudication → conflict analysis → grounded synthesis with citation-linked answers or justified refusals; CATS (Conflict-Aware Trust-Score) LLM-judge pipeline; 539-query dataset; Qwen SFT: correctness 0.069 → 0.883, behavioural adherence 0.074 → 0.722 — [arXiv abs](https://arxiv.org/abs/2512.16795). Search snippet: "conflict-type prediction remains" hard and SFT favours the "Complementary" class — [search](https://arxiv.org/pdf/2512.16795v1)

#### L. Early desiderata / diagnostic work still used as baselines
- **Resolving Knowledge Conflicts in LLMs** — Wang, Feng, ..., Tsvetkov; COLM 2024; arXiv 2310.00935: three desiderata — identify conflicts, pinpoint conflicting segments, provide distinct answers; LLMs good at the first, poor at the last two; instruction-based approaches proposed (no numbers on abs page) — [arXiv abs](https://arxiv.org/abs/2310.00935)
- **DRAGged into Conflicts** — Cattan et al. (Google Research); arXiv 2506.08500 (Jun 2025): CONFLICTS benchmark with expert conflict-type labels; ~65% of instances contain a conflict; prompting models to explicitly reason about possible conflicts improved response quality but "substantial room" remains; code/data https://github.com/google-research-datasets/rag_conflicts — [arXiv abs v2](https://arxiv.org/abs/2506.08500v2)

### Inferences
- Among methods with released code AND evaluation on an inter-document conflict benchmark, only MADAM-RAG (RAMDocs/AmbigDocs/FaithEval) and SABER (ConflictQA/ConflictBank/ConFiQA) qualify; most other code releases (CAD, AdaCAD, CoCoA, CARE, ProbeRAG, RA-RAG) target context-memory conflict on NQ-Swap/ConFiQA-style or synthetic-source setups.
- Per-query cost ordering (derived from the papers' own statements): CAD/AdaCAD/CoCoA ≈ 2× decode compute, 1 call; CAG 1 call; Astute 2–4 calls; RA-RAG ~5–9 calls; ConflictRAG ≥ (0.27×10 pairs) + ≥3 calls; MADAM-RAG ≈ N_docs agents × ≤3 rounds + aggregator per round (RAMDocs document count not stated in README; derived, not reported by paper).
- MADAM-RAG's gains are largest where the baseline is weakest (FaithEval unanswerable/counterfactual) and smallest on RAMDocs (+1.8 EM over prompt-based for Llama-3.3-70B); the 2026 poisoning re-implementation suggests it is fragile to implementation choices and over-abstains.

### Gaps
- Astute RAG: no official code repository could be located; MADAM-RAG and TCR used re-implementations.
- Re³, TrustPropRAG, FRESCO, T-GRAG: full-text numbers not extracted (PDF-only or abstract-only pages); Re³ code URL not found.
- SABER: baseline rows of Tables 1–2 not extracted; ERA datasets beyond NQ/TriviaQA/WebQ/WikiEvent not confirmed.
- "Resolving Knowledge Conflicts" (Wang 2023) and DRAGged into Conflicts: exact per-model numbers not fetched.
- No 2026 paper was found that reports beating MADAM-RAG on RAMDocs; the only RAMDocs follow-up located is the poisoning comparison (arXiv 2605.05632), which does not evaluate on RAMDocs itself.
- KnowPO, Context-DPO, CANOE, ParamMute, COIECD, KRE, Parenting, IRCAN, RAAT, KAFT appear only as baselines inside other papers; their own numbers were not fetched.

---

## KQ2. Which standard baselines do papers compare against, and what evaluation protocol is standard?

### Takeaway
The de-facto baseline set is: No-RAG (closed book), vanilla RAG (concatenate top-k), majority vote / self-consistency over passages (USC, MV), Self-RAG, InstructRAG, RobustRAG, GenRead, Self-Route, Speculative RAG, CRAG, and for decoding papers CAD/AdaCAD/COIECD. Metrics are short-form EM or containment accuracy; set-valued strict EM for multi-answer datasets; and GPT-4-class LLM judges for correctness in 2026 preprints.

### Cited Findings
- Astute RAG baselines: No RAG, RAG, USC (4 calls), GenRead (2), RobustRAG (11), InstructRAG (1), Self-Route (1-2); metric "accuracy" on short-form QA with Gemini 1.5 Pro / Claude 3.5 Sonnet, temperature 0 — [arXiv HTML 2410.07176v1](https://arxiv.org/html/2410.07176v1)
- MADAM-RAG baselines: No RAG, Prompt-based (concatenated), Single Agent with Self-reflection, Speculative RAG, Astute RAG, Self-RAG (SelfRAG-Llama2-13B); strict set EM (all gold, no misinformation-only answers) plus P/R/F1 — [arXiv HTML 2504.13079v2](https://arxiv.org/html/2504.13079v2)
- CAG baselines: retrieval-based, retrieval+reranking, retrieval+credibility-in-prompt for ChatGPT, LLaMA-2-7B/13B/70B, Vicuna-7B, Mistral-7B-Instruct, vanilla IFT; metric EM — [arXiv HTML 2404.06809v3](https://arxiv.org/html/2404.06809v3)
- ConflictRAG baselines: Standard RAG, RAG+Reranking, Self-RAG, CRAG, NLI-Filter, CoT Detection; metrics: Correctness by GPT-4o judge, token F1, CARS — [arXiv HTML 2605.17301v1](https://arxiv.org/html/2605.17301v1)
- CARE-RAG baselines: No RAG, RAG, InstructRAG, GenRead, Self-RAG; EM/F1 — [arXiv HTML 2507.01281](https://arxiv.org/html/2507.01281)
- RA-RAG baselines: Oracle WMV, Majority voting, Vanilla RAG, RobustRAG, Self-RAG, Vanilla LLM; accuracy — [arXiv HTML 2410.22954v5](https://arxiv.org/html/2410.22954v5)
- QACC prompting baselines: Context, Majority Vote, Discern&Answer, Explain&Answer; EM/F1 split by conflicting vs non-conflicting — [arXiv HTML 2410.12311v3](https://arxiv.org/html/2410.12311v3)
- Decoding-family baselines: Greedy, CAD, COIECD, AdaCAD (CoCoA) — [arXiv HTML 2508.17670v1](https://arxiv.org/html/2508.17670v1); plus Opin(Instr), KRE, FaithfulRAG, Context-DPO, CANOE, ParamMute (ProbeRAG) — [arXiv HTML 2510.12460v2](https://arxiv.org/html/2510.12460v2)
- CARE soft-prompt baselines: Direct FT, RetRobust, Closed-book, RAG, CAD, ADACAD, Adaptive-RAG, SKR-kNN, Priori Judgment; Span EM, F1/ROUGE-L, Acc — [arXiv HTML 2508.15253v2](https://arxiv.org/html/2508.15253v2)
- WikiContradict protocol: human eval of 48 instances × 5 models (1,200 samples), then few-shot Llama-3-70b-instruct judge (F-score 80.4 on "correct"; GPT-4 82.5 but closed) over 1,500+ answers; "correct" = answer reflects both conflicting facts — [arXiv HTML 2406.13805](https://arxiv.org/html/2406.13805)
- ConfRAG (ACL 2026) protocol: 1,814 real questions, avg 9.58 paragraphs, 57.2% with explicit contradictions; tasks: answer clustering, answer coverage, reason coverage — [ACL Anthology](https://aclanthology.org/2026.acl-long.11/)
- NATCONFQA (Nachshoni, Cattan, ..., Dagan; arXiv 2508.12355, Aug 2025): extends multi-answer QA so models must return all valid answers *and* detect conflicting answer pairs with conflict-type labels; 8 LLMs evaluated — [arXiv abs](https://arxiv.org/abs/2508.12355)
- Poisoning-study protocol (2026): GPT-5-mini judge, attack success rate with 95% bootstrap CIs, clean-conditioned ASR, behavioural taxonomy (contradiction detection, hedging, unknown) — [arXiv HTML 2605.05632](https://arxiv.org/html/2605.05632)
- Survey taxonomy of mitigation for inter-context conflict: eliminating conflict (specialised NLI/contradiction models: PCNN; general: FacTool, LLM-predicted credibility signals) vs improving robustness (training discriminator: Hong et al.; query augmentation: Weller et al.) — Xu et al., EMNLP 2024, [arXiv HTML 2403.08319](https://arxiv.org/html/2403.08319)

### Inferences
- "Majority vote over passages" is used as a baseline in RA-RAG (MV), QACC (Majority Vote), Astute (USC = self-consistency); it is competitive on real conflicts (QACC Phi-3: 52.52 vs 49.57 context) but collapses under adversarial source imbalance (RA-RAG NQ: 0.327 at 7/9 adversaries).
- RankRAG and CRAG are rarely used as conflict baselines: CRAG appears only in ConflictRAG (54.1 on ConflictQA) and as the named baseline in EvidentialRAG; no paper in this set uses RankRAG.
- Exact match vs containment: Astute RAG and RA-RAG use "accuracy" (containment-style); MADAM-RAG, CAG, CARE-RAG, Discern-and-Answer, CoCoA use EM; 2026 preprints (ConflictRAG, poisoning study, CATS) shift to LLM judges — results across these families are not directly comparable.

### Gaps
- No paper in this set compares against RankRAG; InstructRAG appears in Astute, CARE-RAG, TCR only.
- Self-RAG is consistently weak as a conflict baseline (ConflictRAG 47.8 < Standard RAG 49.2; RAMDocs 26.80), but no paper analyses why.

---

## KQ3. Which methods explicitly output "unresolved" / "multiple valid answers", and how is that evaluated?

### Takeaway
Only a handful of methods are designed to return a set of answers or an explicit non-answer: MADAM-RAG (all disambiguated answers; strict set-EM), ConflictRAG (multi-perspective synthesis for opinion conflicts; confidence qualifier; no explicit "unresolved"), the three belief-based abstainers (ERA, SABER, EvidentialRAG) and the reasoning-trace SFT framework (justified refusals). Evaluation is either set-level strict EM (RAMDocs/AmbigDocs), "mentions both facts" judged by humans/LLM (WikiContradict), answer-clustering/coverage (ConfRAG, NATCONFQA), or risk–coverage / abstention-F1 curves (ERA, SABER).

### Cited Findings
- MADAM-RAG: aggregator "collates responses for disambiguated entities while discarding misinformation and noise"; correct iff output "includes all gold answers and no incorrect or misleading answers" — [arXiv HTML 2504.13079v2](https://arxiv.org/html/2504.13079v2); RAMDocs instances carry `gold_answers`, `wrong_answers`, `disambig_entity`, and per-document `type` ∈ {correct, misinfo, noise} — [GitHub](https://github.com/HanNight/RAMDocs)
- AmbigDocs defines "an ontology of four types of incomplete answers and automatic metrics to identify them"; SOTA models "often give ambiguous answers or incorrectly merge information belonging to different entities" — [arXiv abs 2404.12447](https://arxiv.org/abs/2404.12447)
- ConflictRAG: "there is no explicit 'unresolved' output. Multiple viewpoints are presented for opinion conflicts, and the answer includes conflict annotations and a confidence qualifier"; opinion conflicts are the hardest category (60.4% correctness) — [arXiv HTML 2605.17301v1](https://arxiv.org/html/2605.17301v1)
- WikiContradict: with two contradictory passages, human-judged "correct" (reflects both facts) under neutral prompt (template 4): Mistral-7b 2.1%, Mixtral-8x7b 4.2%, Llama-2-70b-chat 4.2%, Llama-3-70b 10.4%, GPT-4-turbo 6.3%; with mitigation prompt (template 5, "consider conflicting information"): 20.8%, 14.6%, 22.9%, 43.8%, 10.4%. Llama-3-70b template 5: 60.0% on explicit vs 17.6% on implicit conflicts. GPT-4 flagged contradictions ~88% when asked directly, but confirmed "no conflict" on conflict-free passages only 44% — [arXiv HTML 2406.13805](https://arxiv.org/html/2406.13805)
- ERA reports Answer F1 / Abstention F1 / Overall F1 (Qwen3-8B: ERA answer F1 0.573, abstention F1 0.520, overall 0.545) — [arXiv HTML 2604.20854](https://arxiv.org/html/2604.20854)
- SABER: 4-cell decision including abstain; "risk-coverage curve Pareto-dominates every prompt-based abstainer" — [arXiv abs 2605.18792](https://arxiv.org/abs/2605.18792)
- EvidentialRAG: fused uncertainty routes to "direct answering, conflict-aware answering, or abstention"; conflict resolution 35.2% → 51.2% on CRAG ambiguous subset — [arXiv abs 2607.10491](https://arxiv.org/abs/2607.10491)
- From Facts to Conclusions: outputs "citation-linked answers or justified refusals"; CATS scores refusal accuracy and conflict-behaviour alignment — [arXiv abs 2512.16795](https://arxiv.org/abs/2512.16795)
- Poisoning study measured MADAM-RAG hedging 52.2% and "unknown" 19.4% under attack, vs vanilla 10.9%/0.1% — [arXiv HTML 2605.05632](https://arxiv.org/html/2605.05632)
- Wang et al. 2023 desideratum 3: "provide distinct answers or viewpoints"; LLMs "struggle ... to produce responses with distinct answers" — [arXiv abs 2310.00935](https://arxiv.org/abs/2310.00935)
- ConfRAG tasks "answer clustering, answer coverage, and reason coverage" measure organising contradictory answers and explaining conflicts — [ACL Anthology 2026.acl-long.11](https://aclanthology.org/2026.acl-long.11/)

### Inferences
- There is no shared metric for "unresolved" outputs: set-EM (RAMDocs) penalises any extra wrong answer, WikiContradict rewards mentioning both, and abstention papers use risk–coverage; a team must pick one and report it alongside plain accuracy.
- Over-abstention is a real failure mode (MADAM-RAG 41.4% non-answers on clean NQ in the 2026 re-implementation), so any "unresolved" output should be scored with a coverage penalty.

### Gaps
- AmbigDocs' four incomplete-answer categories and metric definitions were not extracted (abs page only).
- No method paper was found that is evaluated on NATCONFQA or ConfRAG yet.

---

## KQ4. Which methods use document metadata (date, source type, authority, version) and how are the signals obtained?

### Takeaway
Three patterns exist: (1) hand/pipeline rules on real metadata — CAG uses publication date vs an LLM-voted validity period and hand-assigned per-scenario source reliability; (2) LLM-inferred signals — ConflictRAG extracts authority/recency/relevance/specificity/consistency with an LLM and weights them by entropy; (3) learned/estimated reliability without metadata — RA-RAG estimates per-source reliability from cross-source agreement, TrustPropRAG propagates sparse human reliability feedback over a document graph, Re³ encodes timestamps inside the retriever. Most conflict methods (MADAM-RAG, Astute, decoding methods) use no metadata at all.

### Cited Findings
- CAG: relevance from SPLADE score bins; timeliness from "temporal gap between the query time and the document's publication date", validity periods "chosen by prompting an LLM three times and voting"; source reliability "set per scenario" (LLM fake news low, reputable sites high); final credibility = min(adjusted relevance, source) — [arXiv HTML 2404.06809v3](https://arxiv.org/html/2404.06809v3)
- ConflictRAG: "Five LLM-extracted criteria: authority, recency, relevance, specificity, and consistency. Weights are derived from entropy"; authority 0.312 and recency 0.245 are most discriminating; recency from "metadata or LLM-extracted dates"; ablating consistency costs only 2.1% selection accuracy; the authors warn consistency "may favor an incorrect majority" — [arXiv HTML 2605.17301v1](https://arxiv.org/html/2605.17301v1)
- RA-RAG: reliability estimated "without labels by iterating between estimated source reliability and estimated true answers across a set of queries"; the authors criticise credibility methods that rely on the LLM's internal knowledge as misaligned "with RAG's core rationale of leveraging external knowledge" — [arXiv abs 2410.22954](https://arxiv.org/abs/2410.22954); [search summary](https://arxiv.org/html/2410.22954v4)
- TrustPropRAG: initial trust from "a limited set of human reliability feedback", propagated multi-hop over a document-relation graph via an optimisation problem — [ACL Anthology 2026.findings-emnlp.711](https://aclanthology.org/2026.findings-emnlp.711/)
- Re³: "Time-Aware Dual Relevance Encoder that folds different kinds of temporal signals into the semantic embedding space" + list-wise Conflict-Aware Recency Filter — [ACL Anthology 2026.acl-long.1180](https://aclanthology.org/2026.acl-long.1180/)
- FRESCO: re-rankers favour "older, semantically rich documents" even when obsolete; fix is instruction optimisation of the reranker, not metadata — [arXiv abs 2604.14227](https://arxiv.org/abs/2604.14227)
- Astute RAG tracks only *source = internal vs external* ("source-awareness") — [arXiv HTML 2410.07176v1](https://arxiv.org/html/2410.07176v1)
- Leiden thesis (secondary): "metadata of source documents can influence LLMs' trust" and attribution — [search result](https://scholarlypublications.universiteitleiden.nl/access/item%3A4296736/view) (not fetched)

### Inferences
- No surveyed method learns a classifier over real-world metadata (publisher, version, authority) from labelled data; signals are either hand rules (CAG), LLM extraction (ConflictRAG), or label-free statistical estimation (RA-RAG). A learned metadata-credibility model would be novel relative to this set.
- CAG's credibility scheme is the most directly reusable for a corpus with timestamps and source types, but its labels are scenario-specific and partially manual.

### Gaps
- How CAG assigns credibility at inference time for new corpora is "not fully specified" in the paper — [arXiv HTML 2404.06809v3](https://arxiv.org/html/2404.06809v3).
- ConflictRAG's extraction prompts are in a supplementary file not included in the arXiv text.

---

## KQ5. Typical reported gains over vanilla RAG on RAMDocs, AmbigDocs, WikiContradict, ConflictBank, FaithEval, Discern-and-Answer/MacNoise, ConflictQA

### Takeaway
Gains are small on realistic inter-document conflict (RAMDocs +1.8 to +5.8 EM) and large on synthetic context-memory conflict (FaithEval +15.8 EM for MADAM-RAG; ProbeRAG FaithEval EM 53.1 → 64.4; ConFiQA-QA EM 49.0 → 91.7). WikiContradict and ConflictBank have essentially no published *method* numbers beyond prompt variants and SABER.

### Cited Findings
- RAMDocs (EM): prompt-based → MADAM-RAG: Llama-3.3-70B 32.60 → 34.40; Qwen2.5-72B 20.60 → 26.40; GPT-4o-mini 25.00 → 28.00; Astute RAG 31.80 / 20.80 / 13.00 — [arXiv HTML 2504.13079v2](https://arxiv.org/html/2504.13079v2)
- AmbigDocs (EM): prompt-based → MADAM-RAG: 54.20 → 58.20; 41.20 → 52.70; 51.50 → 63.00; Self-RAG 55.00; Astute 46.80 / 39.80 / 15.00 — [arXiv HTML 2504.13079v2](https://arxiv.org/html/2504.13079v2)
- FaithEval (EM): prompt-based → MADAM-RAG: 27.30 → 43.10; 38.50 → 57.70; 21.00 → 38.50; Speculative RAG 41.80 / 56.20 / 36.50 — [arXiv HTML 2504.13079v2](https://arxiv.org/html/2504.13079v2). ProbeRAG FaithEval F1/EM, LLaMA-3.1-8B-Inst: Full-Ctx 66.9/53.1; CAD 59.4/42.7; AdaCAD 61.4/47.7; Context-DPO 67.2/53.7; CANOE 71.6/56.3; ParamMute 68.5/56.2; ProbeRAG 74.4/64.4 — [arXiv HTML 2510.12460v2](https://arxiv.org/html/2510.12460v2)
- WikiContradict: no method paper found; best reported is prompt mitigation, human-judged correct 10.4% → 43.8% (Llama-3-70b); automatic judge correct rates (template 5): Mistral-7b 50.6, Llama-3-70b 45.8, GPT-4 15.0 (authors suspect judge over-credits Mistral for listing both answers) — [arXiv HTML 2406.13805](https://arxiv.org/html/2406.13805)
- ConflictBank: 7,453,853 claim-evidence pairs, 553,117 QA pairs; misinformation/temporal/semantic conflicts; 12 LLMs — [arXiv abs 2408.12076](https://arxiv.org/abs/2408.12076). Only method numbers found: SABER Acc/MFS 80.2/92.5 (Llama-3.1-8B), 70.9/89.9 (Llama-3.2-3B), 73.5/86.0 (Qwen-2.5-7B), 71.6/85.4 (Qwen-2.5-3B) — [arXiv HTML 2605.18792v1](https://arxiv.org/html/2605.18792v1); TCR lists "ConflictBank 2024" among datasets but no per-dataset table was extracted — [arXiv HTML 2601.06842](https://arxiv.org/html/2601.06842)
- Discern-and-Answer / MacNoise (dev EM avg over 0–35%): FiD 45.2 → 46.5 (NQ), 50.2 → 55.8 (TQA); GPT-3.5 29.7 → 33.3 (prompt disc) → 35.0 (FiD disc) on NQ; 57.3 → 59.9 → 64.7 on TQA — [arXiv HTML 2305.01579v3](https://arxiv.org/html/2305.01579v3)
- ConflictQA (Xie et al., ICLR 2024 spotlight; PopQA/StrategyQA with LLM-generated counter-memory; code https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict) — [arXiv abs 2305.13300](https://arxiv.org/abs/2305.13300). Method numbers: ConflictRAG correctness 49.2 (Standard RAG) → 68.9 (GPT-4o-mini, LLM-judge) — [arXiv HTML 2605.17301v1](https://arxiv.org/html/2605.17301v1); SABER ConflictQA-PopQA Acc/MFS 53.5/91.0 (Llama-3.1-8B) — [arXiv HTML 2605.18792v1](https://arxiv.org/html/2605.18792v1); EvidentialRAG evaluated on ConflictQA but reports numbers only for CRAG — [arXiv abs 2607.10491](https://arxiv.org/abs/2607.10491)
- ConFiQA (context-memory, counterfactual): ProbeRAG QA EM 91.7 vs Full-Ctx 49.0 (LLaMA-3.1-8B) — [arXiv HTML 2510.12460v2](https://arxiv.org/html/2510.12460v2); SABER 58.4/82.5 — [arXiv HTML 2605.18792v1](https://arxiv.org/html/2605.18792v1)
- NQ-SWAP (context-memory): CAD LLaMA-30B 9.6 → 37.7; CoCoA Llama3-70B greedy 55.74 → 88.32 — [arXiv HTML 2305.14739v1](https://arxiv.org/html/2305.14739v1); [arXiv HTML 2508.17670v1](https://arxiv.org/html/2508.17670v1)
- Real conflicting web contexts (QACC, Phi-3-14B): few-shot 49.57 → FT+Explain&Answer 54.74 EM total; GPT-4o gets *no* gain from any prompt variant (Context 51.17 is best) — [arXiv HTML 2410.12311v3](https://arxiv.org/html/2410.12311v3)
- Astute RAG on imperfect real retrieval (Claude 3.5 Sonnet overall): RAG 55.47 → Astute 61.71–62.86 — [arXiv HTML 2410.07176v1](https://arxiv.org/html/2410.07176v1)

### Inferences
- A team targeting RAMDocs should treat ~34 EM (Llama-3.3-70B) / 28 EM (GPT-4o-mini) as the published SOTA; for AmbigDocs ~58–63 EM; for FaithEval 43–58 EM depending on model.
- The gap between synthetic context-memory gains (+30–40 EM) and real inter-document gains (+2–6 EM) is the single clearest pattern in the literature.

### Gaps
- No peer-reviewed method numbers on WikiContradict or ConflictBank beyond SABER (preprint) and prompt baselines.
- ConflictQA numbers from ConflictRAG rely on an LLM judge and self-estimated judge bias; not directly comparable with EM.

---

## KQ6. Which methods have released code a team could run as a baseline in two to three weeks?

### Takeaway
Runnable with public code and documented entry points: MADAM-RAG (MIT; one script, HF Llama-3.3-70B), CAG (code+models+benchmark), CAD/AdaCAD/CoCoA (decoding wrappers), CARE soft-prompting (needs ~28 h training on 2×A100), RA-RAG, ProbeRAG, FaithfulRAG, SABER, Discern-and-Answer (FiD-era code). Astute RAG, ConflictRAG, CARE-RAG, TCR, Re³, TrustPropRAG, EvidentialRAG have no located official code.

### Cited Findings
- MADAM-RAG: https://github.com/HanNight/RAMDocs — MIT; `run_madam_rag.py --model_name meta-llama/Llama-3.3-70B-Instruct --dataset_path RAMDocs_test.jsonl`, `num_rounds` arg; README flags: arg named `data_path` in docs vs `--dataset_path` in example; no baselines included — [GitHub](https://github.com/HanNight/RAMDocs)
- CAG: https://github.com/panruotong/CAG — "code, benchmark, and models" (LLaMA-2-7B/13B, Mistral-7B fine-tunes); credibility labels high/medium/low per document; license not stated — [arXiv HTML 2404.06809v3](https://arxiv.org/html/2404.06809v3); [GitHub](https://github.com/panruotong/CAG)
- CAD: https://github.com/xhan77/context-aware-decoding — [arXiv abs 2305.14739](https://arxiv.org/abs/2305.14739v1); AdaCAD: https://github.com/HanNight/AdaCAD — [arXiv HTML 2409.07394v2](https://arxiv.org/html/2409.07394v2); CoCoA: https://github.com/infusion-zero-edit/CoCoA/ — [arXiv HTML 2508.17670v1](https://arxiv.org/html/2508.17670v1)
- CARE soft prompting: https://github.com/eunseongc/CARE; training 2M-context reconstruction pretraining ~25 h + NQ FT ~3 h on 2×A100-80GB — [arXiv HTML 2508.15253v2](https://arxiv.org/html/2508.15253v2)
- RA-RAG: https://github.com/ml-postech/RA-RAG; offline reliability estimation 4.65 min for 5 sources (GPT-4o-mini) — [arXiv HTML 2410.22954v5](https://arxiv.org/html/2410.22954v5)
- ProbeRAG: https://github.com/LinfengGao/ProbeRAG (requires white-box access to attention heads) — [arXiv HTML 2510.12460v2](https://arxiv.org/html/2510.12460v2)
- FaithfulRAG: https://github.com/DeepLearnXMU/Faithful-RAG — [arXiv abs 2506.08938](https://arxiv.org/abs/2506.08938)
- SABER: https://github.com/xizhu1022/SABER; no LLM fine-tuning; ~7 forward passes + self-eval passes per query — [arXiv HTML 2605.18792v1](https://arxiv.org/html/2605.18792v1)
- Discern-and-Answer: https://github.com/wjdghks950/Discern-and-Answer (FiD/T5-base + text-davinci-003 prompts; the GPT-3.5 model is deprecated) — [arXiv HTML 2305.01579v3](https://arxiv.org/html/2305.01579v3)
- ERA: anonymous repo https://anonymous.4open.science/r/ERA-318B — [arXiv HTML 2604.20854](https://arxiv.org/html/2604.20854)
- Poisoning benchmark including a MADAM-RAG re-implementation: https://github.com/samkorn/rag-poisoning-architecture-bench, data https://doi.org/10.5281/zenodo.19582217 — [arXiv HTML 2605.05632](https://arxiv.org/html/2605.05632)
- CONFLICTS data/code (benchmark + prompting): https://github.com/google-research-datasets/rag_conflicts — [arXiv abs 2506.08500v2](https://arxiv.org/abs/2506.08500v2)
- No code: Astute RAG (no repo found), ConflictRAG ("will be released upon acceptance"), CARE-RAG (none in paper), TCR (none), Re³ (none on Anthology page), TrustPropRAG (none), EvidentialRAG (none), FRESCO (none) — see respective citations in KQ1.

### Inferences
- Cheapest credible baseline trio for a 2–3 week sprint: vanilla concat RAG + majority vote (trivial), Astute RAG (re-implement from the 3-step description, 2–4 calls/query; MADAM-RAG authors did exactly this), MADAM-RAG (official script). Add CAD/AdaCAD if the team has logit access, and CAG if they can fine-tune a 7B model.
- Licenses are mostly unstated on paper pages; only RAMDocs (MIT) was confirmed. Teams should check each repo's LICENSE file.

### Gaps
- Licenses for CAG, CAD, AdaCAD, CoCoA, CARE, RA-RAG, ProbeRAG, SABER repos were not verified.
- Whether the QACC dataset/code was actually released with the NAACL 2025 Findings version is unverified (papers.cool says yes; arXiv v3 says "will release").

---

## KQ7. Recognised open problems and criticisms

### Takeaway
The literature itself names: synthetic/unrealistic conflicts, over-reliance on QA benchmarks, majority/popularity and position heuristics, stylistic persuasion by coherent misinformation, inability to pinpoint or present distinct answers, over-abstention and judge bias in newer pipelines, and fragility of multi-agent methods to implementation details and evidence imbalance.

### Cited Findings
- Survey (EMNLP 2024): "Most conflicts are artificially constructed. Real retrieval-augmented settings, such as conflicting Google search results, need more study"; "No single solution fits all" — solutions should depend on query type, conflict source (misinformation, outdated, partial truths), and whether users want one answer or multiple perspectives; "Research overly relies on QA benchmarks"; interplay among conflict types, explainability, multilinguality, multimodality under-studied — [arXiv HTML 2403.08319](https://arxiv.org/html/2403.08319)
- Liu & Roth "Conflicts in Texts" (Findings EMNLP 2025, arXiv 2504.19472): calls for "conflict-aware frameworks for data collection, training, and model use, and for reasoning and retrieval that can reconcile conflicting information"; treats ambiguity as a root cause of factual conflict — [arXiv abs](https://arxiv.org/abs/2504.19472)
- Xie et al. (ICLR 2024): LLMs "highly receptive to external evidence that conflicts with their parametric memory, provided the evidence is coherent and convincing"; strong confirmation bias toward memory-consistent evidence when mixed — [arXiv abs 2305.13300](https://arxiv.org/abs/2305.13300)
- Wang et al. (COLM 2024): models identify that a conflict exists but "struggle to pinpoint the specific conflicting knowledge and to produce responses with distinct answers" — [arXiv abs 2310.00935](https://arxiv.org/abs/2310.00935)
- QACC: prior work "mostly used synthetic conflicts or ambiguous questions"; ~25% of *unambiguous* real questions have conflicting Google results; discern-style prompting hurt GPT-4o/Claude-3 — [arXiv HTML 2410.12311v3](https://arxiv.org/html/2410.12311v3)
- GroupQA (arXiv 2601.06189, Jan 2026): 1,635 controversial questions, 15,058 stance-annotated documents; "Paraphrasing an argument can be more persuasive than providing distinct, independent support"; models favour first-presented evidence; larger models increasingly resist updating; explanations "unfaithful"; LLMs are "vulnerable heuristic followers" — [arXiv abs](https://arxiv.org/abs/2601.06189)
- MADAM-RAG authors: "a substantial gap remains" as imbalance between supporting evidence and misinformation grows; Astute RAG and concat each lose 46% as misinformation increases — [arXiv abs 2504.13079v2](https://arxiv.org/abs/2504.13079v2); [arXiv HTML v2](https://arxiv.org/html/2504.13079v2)
- Poisoning re-implementation: MADAM-RAG poison-conditioned ASR ~50%, "so the aggregator often fails to resolve contradictions"; judge contradiction-detection precision only ~48.5%, raw rates are upper bounds — [arXiv HTML 2605.05632](https://arxiv.org/html/2605.05632)
- ConflictRAG self-reported: LLM judge "may favor structured output" (~2.5%); self-constructed NQ-Conflict may carry "injection artifacts"; CARS metric "partly favors systems with explicit conflict modules"; consistency criterion "may favor an incorrect majority" — [arXiv HTML 2605.17301v1](https://arxiv.org/html/2605.17301v1)
- FRESCO: re-rankers systematically prefer older, semantically rich but obsolete passages — [arXiv abs 2604.14227](https://arxiv.org/abs/2604.14227); When Facts Change: mutability prompting raises temporal talk without accuracy gains — [ACL Anthology 2026.findings-acl.103](https://aclanthology.org/2026.findings-acl.103/)
- Hidden Commitment (AI4Good @ ICML 2026, arXiv 2610.38296): models give one-sided answers even when prompted to disclose conflict; a residual "source-choice" direction tracks which document is used; disclosure is "only partially steerable" (Qwen 0.8B, Gemma 3 1B) — [arXiv abs](https://arxiv.org/abs/2610.38296)
- WikiContradict: implicit conflicts far harder than explicit (Llama-3-70b 17.6% vs 60.0%); models over-detect conflict in conflict-free passages (GPT-4 "no conflict" only 44%) — [arXiv HTML 2406.13805](https://arxiv.org/html/2406.13805)
- RA-RAG limitations: reliability estimation hard for niche topics with few cross-references; offline estimates go stale; a reliable source can be hijacked — [arXiv HTML 2410.22954v5](https://arxiv.org/html/2410.22954v5)
- Decoding methods over-correct in no-conflict settings (CAD Llama3-8B TriviaQA 85.70 → 40.20) — [arXiv HTML 2508.17670v1](https://arxiv.org/html/2508.17670v1)
- Hong et al.: MacNoise is LLM-generated and "may still contain artifacts"; discriminator "does not fully suppress counterfactual influence" — [arXiv HTML 2305.01579v3](https://arxiv.org/html/2305.01579v3)
- M3MAD-Bench (arXiv 2601.02854): multi-agent debate "can sometimes be harmful rather than helpful" — [search snippet](https://arxiv.org/pdf/2601.02854) (not fetched)

### Inferences
- The two most-cited criticisms a new method must answer: (1) evaluation on real, metadata-bearing conflicts rather than LLM-injected counterfactuals; (2) robustness to *imbalanced* and *repeated/paraphrased* misinformation, where majority-style aggregation (including debate) degrades.
- Several 2026 preprints (TCR, ConflictRAG, ERA, Hidden Commitment) carry internal inconsistencies or unusual metadata; a report should mark them as unverified/preprint-quality.

### Gaps
- No single paper quantifies "surface style" persuasion in RAG aggregation directly; the closest evidence is GroupQA (paraphrase > independent support) and Xie et al. (coherence drives receptiveness).
- M3MAD-Bench and GroupQA exact numbers were not extracted.
