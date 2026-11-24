# Public datasets and benchmarks for RAG / LLM QA under conflicting retrieved evidence (inter-context knowledge conflicts), 2023 – Nov 2026

Research date: 2026-11-14. All figures below were read from the primary paper (arXiv abstract/HTML), the official GitHub/HuggingFace page, or the conference PDF unless stated otherwise. Items marked **UNVERIFIED** could not be confirmed against a primary source and should be re-checked before use.

Decision vocabulary used throughout for the pipeline under evaluation: `answered`, `unresolved_conflict`, `needs_information`, `insufficient_evidence`.

---

## KQ0. Which datasets exist, and which ones are the best fit for a pipeline that (a) answers under conflict, (b) must pick an authoritative document, and (c) may legitimately abstain / report an unresolved conflict?

### Takeaway
Roughly 30 relevant public resources exist (2021 – Sept 2026). Only a handful make "report the conflict" or "abstain" the *scored-correct* behaviour (WikiContradict, FaithEval-Inconsistent, CONFLICTS, Ragability, LIT-RAGBench, WhoQA/AmbigDocs via all-answers recall); only a handful carry per-document gold/misinformation/noise/outdated labels (RAMDocs, MacNoise, RAGuard, CONFACT, HoH, QACC attribution, ConflictBank, ClashEval, ConflictingQA stance); and only the 2025–26 temporal/version sets (CONFLICTS-Outdated, HoH, Stale-Document Poisoning, VersionQA, EnterpriseRAG-Bench Conflicting-Info, Re²Bench) test "newer/authoritative source wins".

### Cited Findings
- The 2024 EMNLP survey "Knowledge Conflicts for LLMs: A Survey" (Xu, Qi, Guo, Wang, Wang, Zhang, Xu; EMNLP 2024 main, pp. 8541–8565) defines three conflict classes — context-memory, inter-context, intra-memory — and its Table 1 lists only five inter-context datasets as of mid-2024: WikiContradiction (Hsu et al. 2021, 2,210), ClaimDiff (Ko et al. 2022, 2,941), Pan et al. 2023a (SQuAD-based substitution, 52,189), ContraDoc (Li et al. 2023a, 449) and ConflictingQA (Wan et al. 2024, 238) — [arXiv 2403.08319v2](https://arxiv.org/html/2403.08319v2); [ACL Anthology 2024.emnlp-main.486](https://aclanthology.org/2024.emnlp-main.486); survey repo [github.com/pillowsofwind/Knowledge-Conflicts-Survey](https://github.com/pillowsofwind/Knowledge-Conflicts-Survey)
- No dedicated 2025–2026 survey on inter-context conflict was found; the LREC 2026 Ragability paper's related-work section (May 2026) is the most recent published overview and lists WikiContradict, ECON, Gokul et al. 2025, MAGIC, ConflictBank and MMKC-Bench as the inter-context benchmarks it compares against — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)

### Inferences
- Shortlist for the four-way decision pipeline (my judgement, based on the per-dataset facts in KQ1–KQ5 below):

| Dataset | Year/venue | Size | Doc origin | Conflict type | Per-doc labels | "Conflict/abstain" can be correct? | Fit |
|---|---|---|---|---|---|---|---|
| RAMDocs | COLM 2025 | 500 q, ~5.53 docs/q | Brave-search chunks + entity-substituted misinfo + noise | ambiguity + misinformation + noise, imbalance | `correct`/`misinfo`/`noise` | Partially: must output *all* valid answers, exclude misinfo | Best single multi-conflict set |
| CONFLICTS | arXiv Jun 2025 (Google) | 458 q, ~9.2 Google results/q | real web | 5-type taxonomy incl. outdated, misinformation, conflicting opinions | per-query type only; per-result date | Yes: "Conflicting opinions" ⇒ neutral summary; "Outdated" ⇒ prefer newest | Best real-web taxonomy; small |
| WikiContradict | NeurIPS 2024 D&B | 253 instances, 2 passages | real Wikipedia | explicit/implicit factual contradiction | answer per passage | Yes: correct = both answers + flag contradiction | Best "unresolved_conflict is correct" set |
| FaithEval-Inconsistent | ICLR 2025 | 1.5K | original + GPT-4o-fabricated passage | misinformation-style | none exposed (gap) | Yes: gold answer is literally "conflict" | Clean abstention/conflict metric |
| QACC | NAACL 2025 Findings | 1,510 q, top-10 Google | real web | real conflicts; annotator reasons Majority/Source/Time/Common sense | answer→context attribution | No (single gold answer) | Tests "which doc is authoritative" |
| HoH | ACL 2025 | 111,972 QA | Wikipedia monthly snapshots | outdated vs current | current evidence + `outdated_infos` w/ timestamps | No | Large temporal set |
| EnterpriseRAG-Bench Conflicting Info | arXiv May 2026 | 20 q (of 500), 2 gold docs | synthetic enterprise docs | newer doc supersedes older | implicit (pair tracked) | No | Closest to "policy supersession" |
| Stale-Document Poisoning | arXiv Sep 2026 | 317 reversals | dated official sources (medicine/law/software/policy) | outdated vs current | stale/current w/ dates | No | Closest to legal/policy recency; no data link found |
| Ragability | LREC 2026 | 1,060 generated test instances | WikiContradict implicit pairs w/ fantasy entities | misinformation (categorical/numerical/temporal) | n/a | Yes: `refusal_not_answerable` metric | Only set scoring refusal on *both* missing and contradictory context |
| WhoQA / AmbigDocs | EMNLP 2024 Findings / arXiv 2024 | 5,152 q / 36,098 q | real Wikipedia | same-name entity ambiguity | per-entity contexts | All answers must be listed; clarification = rejection | `needs_information` proxy |
| RAGuard / CONFACT | NeurIPS 2025 / IJCAI 2025 | 2,648 claims / 3,180 claims | Reddit / Google top-10 | misleading evidence / stance+credibility | supporting/misleading/unrelated ; stance + credibility 1–5 | No | Source-credibility selection |

- A pipeline that outputs `unresolved_conflict` will be scored *correct* on WikiContradict T5, FaithEval-Inconsistent, CONFLICTS "Conflicting opinions", Ragability, and LIT-RAGBench "Contradictory Evidence"; it will be scored *wrong* on RAMDocs/AmbigDocs/WhoQA (which require enumerating all disambiguated answers), on QACC (single gold), and on all temporal sets (which require picking the newest).

### Gaps
- No public dataset was found that encodes explicit document **status** metadata (active / draft / superseded) as a field. EnterpriseRAG-Bench tracks superseding pairs internally but the paper does not say whether version/status fields are exposed to the system. Legal/regulatory QA sets found (ObliQA-MP, Bar Exam QA / Housing Statute QA, ClaimRAG-LAW, Isaacus legal-rag-qa) do not test version precedence (see KQ6).
- "ConfliBERT" and "MultiConflict" (named in the brief): no conflicting-evidence dataset under either name was found; ConfliBERT was not investigated further and is **UNVERIFIED** (likely unrelated political-conflict NLP model). "ConflictRAG" exists (May 2026) but is a method paper that releases no dataset (KQ5).

---

## KQ1. Core multi-document inter-context conflict benchmarks (ambiguity + misinformation + noise; real contradictions)

### Takeaway
RAMDocs (built on AmbigDocs) is the only set that mixes ambiguity, misinformation, noise and support-imbalance in one example with per-document labels; WikiContradict, CONFLICTS and QACC are the only ones using *real* (non-synthetic) contradictions, and only WikiContradict/CONFLICTS reward "report the conflict".

### Cited Findings

#### RAMDocs + MADAM-RAG
- Paper: "Retrieval-Augmented Generation with Conflicting Evidence", Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal; COLM 2025; arXiv 2504.13079 (v1 17 Apr 2025, v2 12 Aug 2025) — [arXiv abs](https://arxiv.org/abs/2504.13079)
- Code/data: [github.com/HanNight/RAMDocs](https://github.com/HanNight/RAMDocs) (MIT license); HF dataset [HanNight/RAMDocs](https://huggingface.co/datasets/HanNight/RAMDocs); raw file `RAMDocs_test.jsonl` — [GitHub README](https://github.com/HanNight/RAMDocs)
- Size/construction: 500 queries built on AmbigDocs; 1–3 correct answers per query (mean 2.20); each answer supported by 1–3 chunks (mean 3.84 supporting docs/query); docs from Brave Search API, split into ≤100-word chunks and kept only if they contain the correct answer; 0–2 misinformation docs per query created by replacing the correct entity with a plausible wrong one (procedure similar to Longpre et al. 2021); 0–2 noisy docs (irrelevant or low-quality); mean 5.53 docs/query of which 1.70 are misinformation or noise — [arXiv HTML](https://arxiv.org/html/2504.13079)
- Fields: `question`, `documents` (each with `text`, `type` ∈ {`correct`,`misinfo`,`noise`}, `answer`; noise docs have `answer: unknown`), `disambig_entity`, `gold_answers`, `wrong_answers` — [GitHub README](https://github.com/HanNight/RAMDocs)
- Metric: exact match where a response is correct only if it includes *all* gold answers and no incorrect answer supported only by misinformation — [arXiv HTML](https://arxiv.org/html/2504.13079)
- Main results (EM %), Llama-3.3-70B-Instruct: No RAG 5.80 / Concatenated 32.60 / Astute RAG 31.80 / MADAM-RAG 34.40 on RAMDocs; AmbigDocs 4.30 / 54.20 / 46.80 / 58.20; FaithEval 26.70 / 27.30 / 37.10 / 43.10. Qwen2.5-72B: RAMDocs 4.20 / 20.60 / 20.80 / 26.40; AmbigDocs 1.80 / 41.20 / 39.80 / 52.70; FaithEval 26.40 / 38.50 / 44.60 / 57.70. GPT-4o-mini: RAMDocs 2.50 / 25.00 / 13.00 / 28.00; AmbigDocs 1.00 / 51.50 / 15.00 / 63.00; FaithEval 31.00 / 21.00 / 34.00 / 38.50 — [arXiv HTML](https://arxiv.org/html/2504.13079)
- Imbalance analysis: on 200 two-answer examples, baselines lose up to 8% as support imbalance grows; adding 1–3 misinformation docs drops Concatenated and Astute RAG by 46% each — [arXiv HTML](https://arxiv.org/html/2504.13079)
- The RAMDocs paper uses FaithEval's inconsistent subset as "1,000 instances" with a single correct answer — [arXiv HTML](https://arxiv.org/html/2504.13079) (note: FaithEval's own paper says the subset has 1.5K; see KQ2)

#### AmbigDocs
- Paper: "AmbigDocs: Reasoning across Documents on Different Entities under the Same Name", Yoonsang Lee, Xi Ye, Eunsol Choi; arXiv 2404.12447 (v3, 9 Aug 2024) — [arXiv HTML v3](https://arxiv.org/html/2404.12447v3)
- Size: 36,098 questions (train 25,268 / dev 3,610 / test 7,220; 60/10/30); mean 2.92 answers/question; 102,624 distinct entities; 49.5% of questions have 2 answers, 26.6% have 3, 13.3% have 4, 6.4% have 5, 4.3% >5; gold documents = number of answers; gold+retrieved and retrieved-only settings use 5 documents — [arXiv HTML v3](https://arxiv.org/html/2404.12447v3); [ambigdocs.github.io](https://ambigdocs.github.io/)
- Construction: Wikipedia disambiguation pages → documents about different same-name entities → questions with the ambiguous name — [arXiv abs](https://arxiv.org/abs/2404.12447)
- Answer ontology (5 types): Complete; and four incomplete — Partial (some valid answers with entity names), No Answer (abstains), Ambiguous (one valid answer without entity name), Merged (multiple answers without entity names, mixing entities; "most problematic") — [arXiv HTML v3](https://arxiv.org/html/2404.12447v3)
- Metrics: Answer Recall, Entity Recall, Entity-Answer Recall (EAR), Disambig-F1 (DF1), K-Precision — [arXiv HTML v3](https://arxiv.org/html/2404.12447v3)
- Results (gold-only, zero-shot): GPT-4 EAR 0.473 / DF1 0.289; Mistral-7b 0.465 / 0.248; Llama2-7b 0.372 / 0.219; GPT-3.5 0.234 / 0.180; Llama2-13b 0.244 / 0.178. GPT-4 two-shot: EAR 0.647 / DF1 0.382 — [arXiv HTML v3](https://arxiv.org/html/2404.12447v3)
- Data: HF [yoonsanglee/AmbigDocs](https://huggingface.co/datasets/yoonsanglee/AmbigDocs); code [github.com/lilys012/AmbigDocs](https://github.com/lilys012/AmbigDocs); license not stated on project page — [ambigdocs.github.io](https://ambigdocs.github.io/)

#### WikiContradict
- Paper: "WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia", Yufang Hou, Alessandra Pascale, Javier Carnerero-Cano, Tigran Tchrakian, Radu Marinescu, Elizabeth Daly, Inkit Padhi, Prasanna Sattigeri (IBM Research); NeurIPS 2024 Datasets & Benchmarks Track; arXiv 2406.13805 — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c63819755591ea972f8570beffca6b1b-Abstract.html); [arXiv abs](https://arxiv.org/abs/2406.13805)
- Size/construction: ~1,200 Wikipedia articles with maintenance tags (inconsistent / self-contradictory / contradict-other) collected 26 Feb 2024 → 261 tags → 130 verified → 253 annotated instances; each instance = two decontextualised contradictory passages + explanation + taxonomy labels + ≥1 question with a different answer from each passage; ~30 min annotation per tag; Cohen's κ 0.58–0.88 — [arXiv HTML](https://arxiv.org/html/2406.13805)
- Conflict types: Explicit 161 (64%), Implicit 92 (36%); Yes/No 133 (53%), Wh- 120 (47%) (paper text elsewhere says ~61% wh-questions — internal inconsistency) — [arXiv HTML](https://arxiv.org/html/2406.13805)
- Scoring with two contradictory passages (Templates 4/5): Correct = names all contradictory answers, shows no preference, flags the contradiction; Partially correct = one answer or prefers one; Incorrect = no answer match, or lists both as possible without noting the contradiction — [arXiv HTML](https://arxiv.org/html/2406.13805)
- Human-eval Correct % (Template 4 → Template 5): Mistral-7b-inst 2.1 → 20.8; Mixtral-8x7b 4.2 → 14.6; Llama-2-70b 4.2 → 22.9; Llama-3-70b 10.4 → 43.8; GPT-4 6.3 → 10.4; single-passage (T2) correct 88–98% — [arXiv HTML](https://arxiv.org/html/2406.13805)
- Automatic judge WikiContradictEval (Llama-3-70b few-shot) F-score 80.4 (GPT-4 judge 82.5); on all 253 instances T5 correct: Mistral-7b 50.6, Llama-3-70b 45.8, Mixtral 37.9, GPT-4 15.0 (authors warn the judge over-credits Mistral) — [arXiv HTML](https://arxiv.org/html/2406.13805)
- Data: HF [ibm/Wikipedia_contradict_benchmark](https://huggingface.co/datasets/ibm/Wikipedia_contradict_benchmark), MIT license; fields `context1`, `context2`, merged context, `question`, `answer1`, `answer2`, combined reference, `contradictType` (Explicit/Implicit), `samepassage`, `question_ID`, `WikipediaArticleTitle`, `url` — [HF README](https://huggingface.co/datasets/ibm/Wikipedia_contradict_benchmark/raw/main/README.md)

#### CONFLICTS ("DRAGged into Conflicts")
- Paper: "DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs", Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu (Google Research / Bar-Ilan); arXiv 2506.08500 (v1 10 Jun 2025, v2 15 Jun 2025) — [arXiv HTML v2](https://arxiv.org/html/2506.08500v2); [Google Research listing](https://research.google/pubs/dragged-into-a-conflict-detecting-and-addressing-conflicting-sources-in-retrieval-augmented-llms/)
- Taxonomy and expected behaviour: No conflict → direct answer; Complementary → consolidate; Conflicting opinions/research → neutral summary of viewpoints; Outdated ("Freshness") → prioritise up-to-date, optionally acknowledge outdated; Misinformation → disregard, answer from reliable info — [arXiv HTML v2](https://arxiv.org/html/2506.08500v2)
- Size: 458 queries (18 removed for no relevant sources); labels: No conflict 161, Complementary 115, Conflicting opinions 115, Outdated 62, Misinformation 5; sources: ConflictingQA 162, SituatedQA-Geo 105, FreshQA 95, QACC 55, SituatedQA-Temp 41; top-10 Google results per query, mean 9.2 after filtering; two annotators + reconciliation + third expert — [arXiv HTML v2](https://arxiv.org/html/2506.08500v2)
- Metrics: conflict-type classification accuracy (Gemini 2.5 Flash 65.3, Gemini 2.0 Flash 60.5, GPT-4o 59.2, Gemma 3 27B 53.9, Qwen 2.5 72B 53.1); response quality = expected-behaviour adherence (LLM judge, 0.89 agreement vs humans on 100), answer recall, factual grounding. Adherence: Gemini 2.5 Flash Thinking 76.6 (Pipeline prompt) vs 68.3 (Vanilla); Gemini 2.0 Flash 75.3; GPT-4o 74.9 vs 67.2; Oracle (gold type) 88.6–89.3 — [arXiv HTML v2](https://arxiv.org/html/2506.08500v2)
- Data: [github.com/google-research-datasets/rag_conflicts](https://github.com/google-research-datasets/rag_conflicts), Apache-2.0, single `conflicts.jsonl`; fields `source`, `question`, `search_results` (each `title`, `url`, `snippet`, `date`, `response_str`, `short_text` 512-token window), `conflict_type`, `correct_answer` (only for specific types); no evaluation code; repository is archived/read-only — [GitHub](https://github.com/google-research-datasets/rag_conflicts)

#### QACC (Question Answering with Conflicting Contexts)
- Paper: "Open Domain Question Answering with Conflicting Contexts", Siyi Liu, Qiang Ning, Kishaloy Halder, Wei Xiao, Zheng Qi, Phu Mon Htut, Yi Zhang, Neha Anna John, Bonan Min, Yassine Benajiba, Dan Roth (AWS AI Labs / UPenn); Findings of NAACL 2025 pp. 1838–1854; arXiv 2410.12311 — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3); [ACL Anthology PDF](https://preview.aclanthology.org/moar-dois/2025.findings-naacl.99.pdf)
- Size: 1,510 QAs — train 394 (29% conflicting), dev 303 (19%), test 813 (25%); unambiguous AmbigQA questions + top-10 Google results; MTurk/CROWDAQ annotation (conflict identification, answer→context attribution, correct answer, reason, one-sentence explanation); expert check κ=0.9 on 20 — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3)
- Conflict reasons chosen by annotators: Majority, Source (trustworthiness), Common Sense, Time (most up-to-date); most conflicts are entity mismatches; "How" questions conflict ~40%, "Who" ~20% — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3)
- Metrics EM/F1 on conflicting/non-conflicting/total; best: GPT-4o few-shot EM-T 51.17 / F1-T 67.32; Phi-3 fine-tuned Explain&Answer EM-T 54.74 / F1-T 67.19; Claude-3 few-shot 50.68 / 66.26 — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3)
- Data/code: [github.com/amazon-science/qa-with-conflicting-context](https://github.com/amazon-science/qa-with-conflicting-context), Apache-2.0; `data/ConflictQA_Dataset.json`; EM/F1 eval scripts — [GitHub](https://github.com/amazon-science/qa-with-conflicting-context)
- Naming note: QACC is *not* Chen, Zhang & Choi 2022 "Rich Knowledge Sources Bring Complex Knowledge Conflicts" (EMNLP 2022, pp. 2292–2307), which studies parametric-vs-retrieved conflicts and calibration — [ACL Anthology 2022.emnlp-main.146](https://aclanthology.org/2022.emnlp-main.146)

#### WhoQA
- Paper: "Who's Who: Large Language Models Meet Knowledge Conflicts in Practice", Quang Hieu Pham, Hoang Ngo, Anh Tuan Luu, Dat Quoc Nguyen; EMNLP 2024 Findings; arXiv 2410.15737 — [arXiv HTML](https://arxiv.org/html/2410.15737)
- Size: 5,152 questions over 13 Wikidata property types; 17,365 supporting contexts (~3.4/question; 256-token Contriever-selected Wikipedia passages, two annotators verified); 2–8 distinct answers per question (mean 3.25); ~145,710 entities before filtering — [arXiv HTML](https://arxiv.org/html/2410.15737)
- Scoring: correct only if the response includes *all* answers supported by the contexts; macro-averaged accuracy by number of conflicting answers in three settings (simple QA / conflicts without specification / with specification); "rejection rate" on hard questions — [arXiv HTML](https://arxiv.org/html/2410.15737)
- Results (simple / w/o spec / w/ spec): Llama-3-70B 97.8 / 83.8 / 86.1; Llama-3-8B 98.2 / 72.4 / 71.5; gpt-3.5-turbo 97.9 / 36.0 / 58.9 (rejected 99% of hard questions by asking for clarification); Qwen1.5-72B 95.9 / 28.2 / 66.8; Gemma-7B 91.8 / 12.7 / 34.4 — [arXiv HTML](https://arxiv.org/html/2410.15737)
- Code/data: [github.com/VinAIResearch/WhoQA](https://github.com/VinAIResearch/WhoQA); dataset license not stated — [arXiv HTML](https://arxiv.org/html/2410.15737)

#### MAGIC
- Paper: "MAGIC: A Multi-Hop and Graph-Based Benchmark for Inter-Context Conflicts in Retrieval-Augmented Generation", Jungyeon Lee, Kangmin Lee, Taeuk Kim; Findings of EMNLP 2025; arXiv 2507.21544 (v3 9 Oct 2025) — [arXiv HTML v3](https://arxiv.org/html/2507.21544v3); [ACL Anthology PDF](https://aclanthology.org/2025.findings-emnlp.466.pdf)
- Size/construction: 1,080 context pairs (single-hop 492, multi-hop 588; 1–4 conflicts per pair); Wikidata5M subgraphs (46 relations, 7 domains; ≤15 edges/subgraph); o3-mini generates conflicting triplets with human-filtered demonstrations; GPT-4o-mini verbalises graphs; Claude 3.7 Sonnet checks — [arXiv HTML v3](https://arxiv.org/html/2507.21544v3)
- Tasks/metrics: conflict identification (ID; 0 if missed in any of 3 runs) and localisation of conflicting sentences (LOC; 1 only if all located); manually scored — [arXiv HTML v3](https://arxiv.org/html/2507.21544v3)
- Results: ID GPT-4o-mini 83.61, Llama-3.1-70B 72.86, o1 68.06, Claude 3.5 Haiku 60.28, Mixtral 37.92; LOC GPT-4o-mini 55.00, o1 49.72, Claude 3.5 Haiku 42.50, Llama-3.1-70B 37.92; human pilot 92.5 ID / 83.3 LOC — [arXiv HTML v3](https://arxiv.org/html/2507.21544v3)
- Code: [github.com/HYU-NLP/MAGIC](https://github.com/HYU-NLP/MAGIC); data "released solely for scientific research", no license named — [arXiv HTML v3](https://arxiv.org/html/2507.21544v3)

#### ECon
- Paper: "ECon: On the Detection and Resolution of Evidence Conflicts", Cheng Jiayang et al. (10 authors); EMNLP 2024 main pp. 7816–7844; arXiv 2410.04068 — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.447/); [arXiv HTML v1](https://arxiv.org/html/2410.04068v1)
- Construction: NQ-open and ComplexWebQuestions (answer conflicts), StrategyQA (factoid conflicts, 3–4 factoids); evidence generated by Llama-3-70B-Instruct; "Revise" pollution attack yields lexically-similar conflicts; validation via DeBERTa-v2-xxlarge-mnli NLI + Llama-3-70B consistency check; 200 pairs human-annotated (Fleiss κ 71.2%, pseudo-labels ~92% accurate) — [arXiv HTML v1](https://arxiv.org/html/2410.04068v1)
- Evaluation set sizes: answer-conflict 244 (CWQ-short), 300 (NQ-short), 300 (CWQ-long), 300 (NQ-long); factoid 768 (3 facts), 287 (4 facts) — [arXiv HTML v1](https://arxiv.org/html/2410.04068v1)
- Results: best detector NLI-xxlarge F1 88.8 (short) / 90.9 (long); GPT-4 F1 76.4 / 81.1 (precision 91.8 / 93.9); resolution: Claude 3 Sonnet favoured one side without reasons 23.7%, Claude 3 Haiku 38.1% — [arXiv HTML v1](https://arxiv.org/html/2410.04068v1)
- Code: [github.com/HKUST-KnowComp/EvidenceConflict](https://github.com/HKUST-KnowComp/EvidenceConflict); license not stated — [arXiv HTML v1](https://arxiv.org/html/2410.04068v1)

#### Ragability (LREC 2026)
- Paper: "Ragability Benchmark: A Dataset and Library to Test LLMs on Inter-context Conflicts", Stephanie Gross, Johann Petrak, Brigitte Krenn (OFAI); LREC 2026 pp. 2323–2333 — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)
- Construction: derived from WikiContradict's 92 implicit-reasoning instances (51 distinct context pairs, 8 further removed), with real entities replaced by fantasy entities so parametric memory cannot answer; focuses on misinformation conflicts (categorical, numerical, temporal); library generates many test cases from few entries; 1,060 test instances used in experiments — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)
- Metrics: `correct_answer_all`, `correct_answer_answerable`, `refusal_not_answerable` (correctly refusing when context is missing *or* contradictory), `contradiction_identification` (yes/no) — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)
- Results (7 LLMs): contradiction identification Gemini-2.5-pro 0.9488, gpt-5-mini 0.9434, gpt-5 0.9407, gemini-2.5-flash 0.9380 (mean 0.899); overall `correct_answer_all` ranges 0.7698 (gpt-5) to 0.5302 (llama3) with gpt-5 as checker; `refusal_not_answerable` values 0.3019–0.6469 with gemini-2.5-flash checker; adding a "may contain contradiction" hint raised identification on contradictory contexts but significantly lowered it on non-contradictory ones — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)
- Data/code: HF [ofai/RagabilityCorpus](https://huggingface.co/datasets/ofai/RagabilityCorpus); [github.com/OFAI/python-ragability](https://github.com/OFAI/python-ragability) — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)

### Inferences
- RAMDocs is the only benchmark whose per-document `type` field lets you directly score *document selection* (did the pipeline cite only `correct` docs?) alongside answer EM; its 500-example size and MIT license make it the natural primary testbed, with AmbigDocs test (7,220) for scale on the ambiguity dimension.
- WikiContradict Template 5 is the cleanest existing operationalisation of "`unresolved_conflict` is the correct output" (correct = both answers + flag), but its 253 instances include both genuinely irreconcilable and "one passage is simply wrong" cases without a resolvability label, so an authority-inferring pipeline will be penalised whenever it (correctly) picks one side.
- CONFLICTS is the only dataset whose gold behaviour varies by conflict type in exactly the way the pipeline's decisions do (Outdated → pick newest ≈ `answered` via authority; Conflicting opinions → `unresolved_conflict`; Misinformation → `answered`), but the `correct_answer` field is only populated for some types and Misinformation has just 5 queries.

### Gaps
- AmbigDocs venue (COLM 2024?) not confirmed on arXiv or project page.
- CONFLICTS conference venue not confirmed (arXiv + Google Research listing only).
- WikiContradict dataset card does not list the detailed conflict taxonomy categories; the HF eval-code link is a placeholder.
- Ragability: exact final number of base context pairs truncated in my PDF extraction; dataset license not read.
- MAGIC and WhoQA dataset licenses not stated.

---

## KQ2. Faithfulness / context-conflict sets where "conflict" or "unknown" is the gold answer (FaithEval), and other inconsistent-context resources

### Takeaway
FaithEval-Inconsistent is the only widely used benchmark where the literal gold answer is "conflict"; frontier models already score >93%, so it discriminates mainly among open models, and its fabricated passage is LLM-written rather than real.

### Cited Findings
- Paper: "FaithEval: Can Your Language Model Stay Faithful to Context, Even If 'The Moon is Made of Marshmallows'", Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty (Salesforce); ICLR 2025; arXiv 2410.03727 (v3 24 Apr 2025); paper CC BY-NC-SA 4.0 — [arXiv abs](https://arxiv.org/abs/2410.03727)
- Sizes: Unanswerable 2.4K (~240 per source dataset), Inconsistent 1.5K (150 per source), Counterfactual 1K (ARC-Challenge multiple choice); total 4.9K — [arXiv HTML v3](https://arxiv.org/html/2410.03727v3)
- Inconsistent construction: GPT-4o given question + original answer + original context generates a fabricated new answer and a supporting context; concatenated with the original; sources SQuAD, NewsQA, TriviaQA, NQ, SearchQA, HotpotQA, BioASQ, DROP, RACE, TextbookQA; validated by gpt-4o-mini then majority vote of three MTurk workers — [arXiv HTML v3](https://arxiv.org/html/2410.03727v3)
- Expected answer "conflict"; strict matching accepts only "conflict", non-strict also "conflicting/inconsistent/contradictory/multiple answers"; the two differ by ≤0.001 — [arXiv HTML v3](https://arxiv.org/html/2410.03727v3)
- Inconsistent-context accuracy: Claude 3.5 Sonnet 94.4, GPT-4o 93.2, GPT-4-turbo 85.5, Llama-3.1-70B-Instruct 56.3, Command R+ 51.1, Llama-3-70B 42.9, Llama-3.1-8B 37.2, Phi-3-mini 2.6 — [arXiv HTML v3](https://arxiv.org/html/2410.03727v3)
- Data/code: [github.com/SalesforceAIResearch/FaithEval](https://github.com/SalesforceAIResearch/FaithEval) (Apache-2.0); HF `Salesforce/FaithEval-unanswerable-v1.0`, `Salesforce/FaithEval-inconsistent-v1.0`, `Salesforce/FaithEval-counterfactual-v1.0` (test split); accuracy via normalised phrase match — [GitHub](https://github.com/SalesforceAIResearch/FaithEval)
- Related multimodal set: MMKC-Bench ("Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models", Jia et al., arXiv 2505.19509, later AAAI): 1,573 knowledge instances, 3,381 images, context-memory and inter-context conflicts, LLM counterfactual editing; code [github.com/MLLMKCBENCH/MLLMKC](https://github.com/MLLMKCBENCH/MLLMKC) — [arXiv abs](https://arxiv.org/abs/2505.19509); [AAAI listing](https://ojs.aaai.org/index.php/AAAI/article/view/39385)

### Inferences
- Because the inconsistent context is a single concatenated string (original + fabricated), per-document labels are not exposed as separate fields, so FaithEval scores *only* the `unresolved_conflict` decision, not document selection. Pipelines that infer authority will be penalised here since the gold is always "conflict", never a resolved answer.

### Gaps
- FaithEval HF field schema (whether the fabricated passage is separable) not inspected; the RAMDocs paper's "1,000 instances" vs FaithEval's "1.5K" discrepancy unresolved.

---

## KQ3. Misinformation / counterfactual-noise / parametric-vs-context / source-credibility datasets

### Takeaway
These sets give per-document correct/perturbed or supporting/misleading labels (MacNoise, RAGuard, CONFACT, ClashEval, ConflictQA, KNOWLEDGE CONFLICT) and so support document-selection scoring, but almost all are built by entity substitution or LLM rewriting, and none makes abstention the gold answer (ConflictBank's "uncertain" option excepted).

### Cited Findings

#### Discern-and-Answer / MacNoise ("Why So Gullible?")
- Paper: "Why So Gullible? Enhancing the Robustness of Retrieval-Augmented Models against Counterfactual Noise", Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-Hyon Myaeng, Joyce Jiyoung Whang; NAACL 2024 Findings (not ACL); arXiv 2305.01579 (v3 Jun 2024); paper CC BY-SA 4.0 — [arXiv abs](https://arxiv.org/abs/2305.01579)
- Data: NQ-open (test 3,610; 1,830 perturbable) and TriviaQA-open (test 11,313; 4,464 perturbable); 256 dev sampled for API budget — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Perturbations: (a) entity replacement à la Longpre 2021 (PER/ORG/LOC/DATE/NUM via spaCy); (b) MacNoise — LLM rewrites documents so they no longer support the gold answer (GPT-3.5-turbo for NQ train, GPT-4 for NQ/TQA eval), strategies: rewrite passage, replace entities, negate answer sentence; ≤20 docs perturbed per instance; eval perturbs 0/15/25/35% of retrieved docs; top-5 passages for GPT-3.5 and FiD — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Labels: binary per passage (0 original, 1 perturbed); discriminator trained with BCE + contrastive loss; metrics EM for QA, P/R/F1 for perturbation classification — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Results (EM avg over perturbation levels): FiD entity-replacement NQ test 45.4 → 49.3 with discriminator; GPT-3.5 NQ test 40.1 → 43.9; MacNoise GPT-3.5 dev NQ 29.7 → 35.0, TQA 57.3 → 64.7; discriminator F1 at 35%: 81.00 (entity NQ), 80.62 (MacNoise); MacNoise harder: GPT-3.5 16.0 EM at 35% vs 22.7 entity — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- Code/data: [github.com/wjdghks950/Discern-and-Answer](https://github.com/wjdghks950/Discern-and-Answer) — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)

#### ConflictQA ("Adaptive Chameleon or Stubborn Sloth")
- Paper: Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, Yu Su; ICLR 2024 Spotlight; arXiv 2305.13300 (v4 27 Feb 2024) — [arXiv abs](https://arxiv.org/abs/2305.13300)
- Data: HF [osunlp/ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA), configs `ConflictQA-popQA-[model]` etc.; PopQA and StrategyQA; models ChatGPT, GPT-4, PaLM 2, Llama 2 (7B/70B), Qwen-7B, Vicuna (7B/33B); records hold `memory_answer` + `parametric_memory` and `counter_answer` + generated `counter_memory`; repo Apache-2.0 — [GitHub OSU-NLP-Group/LLM-Knowledge-Conflict](https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict)
- Survey Table 1 lists it at 20,091 examples (PopQA, StrategyQA) plus 9,803 (KC) — [arXiv 2403.08319v2](https://arxiv.org/html/2403.08319v2)
- Findings: LLMs highly receptive to coherent conflicting evidence; strong confirmation bias when some evidence matches parametric memory — [arXiv abs](https://arxiv.org/abs/2305.13300)

#### KNOWLEDGE CONFLICT framework (Wang, Feng et al.)
- Paper: "Resolving Knowledge Conflicts in Large Language Models", Yike Wang, Shangbin Feng et al.; COLM 2024 (per search); arXiv 2310.00935 (v3 15 Oct 2024); CC BY 4.0 — [arXiv HTML v3](https://arxiv.org/html/2310.00935v3); [arXiv abs](https://arxiv.org/abs/2310.00935v2)
- Data: 9,083 entities across 20 domains; conflicts by in-domain entity substitution (3,080) or entity shuffling (6,003); parametric passage from closed-book "context about {entity} in 50 words"; human check 96% (Task 1) / 83.5% (Task 2) clear conflicts, Fleiss κ 0.51 — [arXiv HTML v3](https://arxiv.org/html/2310.00935v3)
- Tasks: (1) contextual conflict detection, (2) QA-span conflict detection, (3) distinct-answers generation; gpt-3.5-turbo: Task 1 F1 0.802 (best baseline 0.783); Task 2 F1 0.535 (Self-ask 0.621); Task 3 acc parametric 0.658 / conflicting 0.815 / both 0.569; GPT-4 CoT both 0.625; fine-tuned Task 1 F1 0.953 — [arXiv HTML v3](https://arxiv.org/html/2310.00935v3)
- Code: [github.com/yikee/Knowledge_Conflict](https://github.com/yikee/Knowledge_Conflict) — [arXiv HTML v3](https://arxiv.org/html/2310.00935v3)

#### ConflictBank
- Paper: "ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLM", Zhaochen Su, Jun Zhang, Xiaoye Qu, Tong Zhu, Yanshu Li, Jiashuo Sun, Juntao Li, Min Zhang, Yu Cheng; NeurIPS 2024 Datasets & Benchmarks (per repo README); arXiv 2408.12076; paper CC BY 4.0 — [arXiv abs](https://arxiv.org/abs/2408.12076); [GitHub](https://github.com/zhaochen0110/conflictbank)
- Construction: Wikidata dump 2 Apr 2024, top-100 relations; misinformation = same-type object replacement; temporal = future time span added (example 20 Apr 2033); semantic = subject description rewritten; evidence generated by LLaMA-3-70B-Instruct in Wikipedia/book/news styles; QC via NLI (DeBERTa-v2-xxlarge-mnli) + SBERT classifier; human check >95% on 200 — [arXiv HTML](https://arxiv.org/html/2408.12076)
- Counts (Table 3): default 1,991,218; misinformation 1,878,340; temporal 1,650,026; semantic 1,934,269 claim-evidence pairs (total 7,453,853); 553,117 QA pairs per type after confirmation — [arXiv HTML](https://arxiv.org/html/2408.12076)
- QA format: multiple-choice A–D including an "uncertain" option; metrics Original Answer Ratio, Counter Answer Ratio, Memorization Ratio = OAR/(OAR+CAR); with only conflicting evidence all models MR <50%; with both default and conflicting evidence all >50% MR; LLaMA2-70B MR 9.45% misinformation / 7.77% temporal / 3.73% semantic; adding descriptions cut LLaMA3-70B MR 67.52% → 8.31% — [arXiv HTML](https://arxiv.org/html/2408.12076)
- Data: HF `Warrieryes/CB_claim_evidence`, `Warrieryes/CB_qa`; repo CC BY-SA 4.0 per README; `inference.py`/`evaluate.py` released — [GitHub](https://github.com/zhaochen0110/conflictbank)

#### ConflictingQA ("What Evidence Do Language Models Find Convincing?")
- Paper: Alexander Wan, Eric Wallace, Dan Klein; ACL 2024 main; arXiv 2402.11782; CC BY 4.0 — [arXiv abs](https://arxiv.org/abs/2402.11782)
- Data: 238 controversial yes/no questions (GPT-4-generated categories → questions; paper inconsistently cites 144 vs 191 categories); 2,208 real web paragraphs (mean 365 words) from Google Search top-20 for affirmative and negative rephrasings, jusText extraction, TAS-B 512-token window; stance label Yes/No by Claude-v1-Instant + GPT-4-1106 agreement — [arXiv HTML](https://arxiv.org/html/2402.11782)
- Metric: paragraph win rate when paired with conflicting paragraphs (≥5 comparisons; 912 paragraphs qualify, mean 6.54 comparisons); finding: relevance drives convincingness, stylistic features (references, neutral tone) do not — [arXiv HTML](https://arxiv.org/html/2402.11782)
- Code: [github.com/AlexWan0/rag-convincingness](https://github.com/AlexWan0/rag-convincingness) — [arXiv HTML](https://arxiv.org/html/2402.11782)

#### ClashEval
- Paper: "ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence", Kevin Wu, Eric Wu, James Zou; NeurIPS 2024 Datasets & Benchmarks; arXiv 2404.10198 (v3 7 Feb 2025); CC BY 4.0 — [arXiv abs](https://arxiv.org/abs/2404.10198); [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2024/file/3aa291abc426d7a29fb08418c1244177-Paper-Datasets_and_Benchmarks_Track.pdf)
- Data: Drug Dosages 249 q ×10 perturbations, Sports Records 191 ×10, Latest News 238 ×10, Wikipedia Dates 200 ×10, Names 200 ×3, Locations 200 ×3 (sum 1,278; paper says "over 1200"); `mod_type` column; metrics accuracy, context bias, prior bias; six models (GPT-4o, GPT-3.5, Llama-3-8b, Gemini 1.5, Claude Opus, Claude Sonnet) — [GitHub kevinwu23/StanfordClashEval](https://github.com/kevinwu23/StanfordClashEval); HF `kewu93/ClashEval`
- Finding: models adopt incorrect retrieved content over correct priors >60% of the time — [arXiv abs](https://arxiv.org/abs/2404.10198)

#### RAGuard
- Paper: "Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals", Linda Zeng, Rithwik Gupta, Divij Motwani, Yi Zhang, Diji Yang; NeurIPS 2025; arXiv 2502.16101 (v5 17 Jan 2026) — [arXiv abs](https://arxiv.org/abs/2502.16101); [NeurIPS 2025 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/ed25c00ff6900989116d3ba5d607d33d-Paper-Datasets_and_Benchmarks_Track.pdf)
- Data: 2,648 PolitiFact political claims (2000–2024, True/False); 16,331 Reddit documents labelled supporting / misleading / unrelated (LLM-generated labels); MIT license; HF [UCSC-IRKM/RAGuard](https://huggingface.co/datasets/UCSC-IRKM/RAGuard)
- Finding: all tested RAG systems performed worse than zero-shot under misleading retrievals; humans do better — [arXiv abs](https://arxiv.org/abs/2502.16101)

#### CONFACT
- Paper: "Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs", Ziyu Ge, Yuhao Wu, Daniel Wai Kit Chin, Roy Ka-Wei Lee, Rui Cao; IJCAI 2025 (AI & Social Good); arXiv 2505.17762 — [arXiv abs](https://arxiv.org/abs/2505.17762); [IJCAI](https://www.ijcai.org/proceedings/2025/1073.pdf)
- Data: 3,180 claims (AVERITEC 566 + FactCheckQA 2,614) → GPT-4 yes/no question → Google top-10 pages archived via Wayback (12–19 Sep 2024); conflict = ≥1 supporting and ≥1 refuting doc per GPT-4; ModC split 611 claims (125 Yes / 486 No), 2,469 sources; HumC (human-validated) 287 claims, 1,418 sources — [arXiv HTML](https://arxiv.org/html/2505.17762)
- Per-document labels: stance (support/refute; 3 GPT-4 prompts + human check), source credibility 1–5, source type (Mainstream News / Government / Non-profit / Academic / Social Media / Other); MBFC ratings as expert ground truth — [arXiv HTML](https://arxiv.org/html/2505.17762)
- Results: without credibility best Qwen-2 + Discern-and-Answer Acc 78.89 / F1 69.54 (ModC); with credibility Mistral Hybrid-MB direct Acc 80.03 / F1 73.46; HumC best with credibility Qwen-2 Acc 82.93 / F1 70.56 — [arXiv HTML](https://arxiv.org/html/2505.17762)
- Code: [github.com/zoeyyes/CONFACT](https://github.com/zoeyyes/CONFACT) (records: `claim`, `label`, `claim_date`, `review_date`, `country`, `question`, evidence items with `original_link`, `content`); license not stated — [GitHub](https://github.com/zoeyyes/CONFACT)

#### Gokul et al. 2025 (Amazon) — contradiction detection in RAG
- "Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency", Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran; arXiv 2504.00180; taxonomy of self-contradictory documents, pairwise contradictions and *conditional* contradictions (document C makes A and B conflict) — [arXiv PDF](https://export.arxiv.org/pdf/2504.00180). Ragability's related work states the dataset is based on HotpotQA with claude-3-sonnet-generated conflicting documents — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf). Data release not checked.

### Inferences
- CONFACT is the only dataset with an explicit *per-document credibility* label (1–5) plus source type, and it demonstrates that credibility signals improve conflict resolution (+1–4 points), making it the best proxy for "infer which document is authoritative" even though its task is claim verification rather than QA.
- RAGuard and MacNoise give misleading/perturbed labels but their gold is always the true answer; a pipeline that outputs `unresolved_conflict` on them will be scored as wrong, so they measure robustness, not calibrated abstention.
- ConflictBank's "uncertain" multiple-choice option is the only abstention affordance in this group, but the paper scores OAR/CAR/MR rather than rewarding "uncertain".

### Gaps
- ConflictQA (Xie) exact per-config sizes not read from HF; ClashEval per-model accuracy/bias numbers are only in Parquet files, not the README; ClashEval license not stated.
- Gokul et al. 2025 dataset size, release and HotpotQA basis not verified from the primary paper.

---

## KQ4. Temporal / outdated-vs-current / document-version conflict datasets

### Takeaway
2025–2026 produced the first datasets where the gold behaviour is "prefer the newer/authoritative version": CONFLICTS-Outdated (62 real-web queries), HoH (111,972 Wikipedia QA with dated outdated versions), Stale-Document Poisoning (317 dated official reversals in medicine/law/software/policy), VersionQA (100 q over 34 versioned technical docs), EnterpriseRAG-Bench Conflicting-Info (20 synthetic superseding pairs) and Re²Bench (1.3M instances, "latest timestamp wins"). Earlier sets (SituatedQA, FreshQA, TAQA) test temporal context, not inter-document conflict.

### Cited Findings

#### HoH
- Paper: "HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation", Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan, Yucong Luo, Jiaying Lin, Qi Liu; ACL 2025; arXiv 2503.04800 (v3 18 Jul 2025) — [arXiv abs](https://arxiv.org/abs/2503.04800); [ACL PDF](https://aclanthology.org/2025.acl-long.301.pdf); [GitHub 0russwest0/HoH](https://github.com/0russwest0/HoH) (Apache-2.0)
- Data: HF [russwest404/HoH-QAs](https://huggingface.co/datasets/russwest404/HoH-QAs): 111,972 QA pairs, single `train` split; monthly English Wikipedia snapshots 2024-06-01 to 2024-12-01; fields `question`, `answer` (current), `evidence`, `last_modified_time`, `outdated_infos` (list of earlier `answer`/`evidence`/`last_modified_time`), `document` (`id`, `title`); Apache-2.0 for authors' contributions, Wikipedia text CC BY-SA 4.0 — [HF dataset](https://huggingface.co/datasets/russwest404/HoH-QAs)
- Finding: outdated information substantially reduces accuracy even when current information is retrieved, and can produce harmful outputs — [arXiv abs](https://arxiv.org/abs/2503.04800)

#### Re³ / Re²Bench (ACL 2026)
- Paper: "Re3: Relevance & Recency Retrieval for Mitigating Temporal Hallucination", Jiawei Cao, Jie Ouyang, Mingyue Cheng, Zhaomeng Zhou, Yupeng Li, Zirui Liu, Chunli Liu, Shijin Wang (USTC / iFLYTEK / HFUT); ACL 2026 Long, pp. 25735–25760; code [github.com/cjwyv/Re3](https://github.com/cjwyv/Re3) — [ACL 2026 PDF](https://aclanthology.org/2026.acl-long.1180.pdf). Earlier arXiv version 2509.01306 (Sep 2025; v2 withdrawn) described Re²Bench as Re²-Rel (TSQA-Human), Re²-Rec (HoH), Re²-Hyb (NOAA) — [arXiv 2509.01306](https://arxiv.org/abs/2509.01306)
- Re²Bench (ACL version): 1.3M instances from three domains — NYC Motor Vehicle Collisions 500K (challenge: relevance), COVID-19 statistics 430K (recency), NOAA weather forecasts 360K (relevance & recency); ground truth defined by latest publication timestamp as winning version; recency vote s_rec(d) ∈ {+1, −1, 0}; metrics Recall@5, MRR, Accuracy — [ACL 2026 PDF](https://aclanthology.org/2026.acl-long.1180.pdf)
- Results: Re³ NYC R@5 0.9248 / MRR 0.8906 / Acc 0.8788; COVID 0.9525 / 0.7215 / 0.9112; NOAA 0.9625 / 0.5929 / 0.8732; on HoH Re³ Acc 0.6339 vs BGE-M3 0.5900; average +9.7% generation accuracy, up to +25.2% on NOAA — [ACL 2026 PDF](https://aclanthology.org/2026.acl-long.1180.pdf)
- Ethics note in paper: the "latest is correct" heuristic can amplify misinformation under data poisoning — [ACL 2026 PDF](https://aclanthology.org/2026.acl-long.1180.pdf)

#### Stale-Document Poisoning
- Paper: "Stale-Document Poisoning: When Outdated Retrieval Overrides Correct Model Answers", Md Shamim Ahmed, Lukas Galke Poech, Richard Röttger; arXiv 2609.31342 (25 Sep 2026); CC BY 4.0 — [arXiv abs](https://arxiv.org/abs/2609.31342)
- Data: 317 verified knowledge reversals in medicine, law, software, platform policy, grounded in dated official sources; outdated retrieval flips 30% (Llama) / 37% (Qwen) of answers without trust instructions, 66% / 75% with; poisoning 17–91% across 4 models × 4 domains; matched up-to-date evidence followed 97–100%; recency-aware hybrid re-ranker reduces poisoning 4.6–10.0 points when dates are accurate; 12 models evaluated — [arXiv abs](https://arxiv.org/abs/2609.31342)
- No GitHub/HF link on the arXiv page — [arXiv abs](https://arxiv.org/abs/2609.31342)

#### FRESCO
- "FRESCO: Benchmarking and Optimizing Re-rankers for Evolving Semantic Conflict in Retrieval-Augmented Generation", Sohyun An, Hayeon Lee, Shuibenyang Yuan, Chun-cheng Jason Chen, Cho-Jui Hsieh, Vijai Mohan, Alexander Min (Meta Superintelligence Labs / UCLA); arXiv 2604.14227 (14 Apr 2026); CC BY 4.0; recency-seeking queries paired with historical Wikipedia revisions; Evolving vs Non-Evolving Knowledge tasks; re-rankers show strong bias to older semantically rich docs; proposed method +27% on Evolving tasks; no repo listed — [arXiv abs](https://arxiv.org/abs/2604.14227). Search snippet reports "Obsolete Ratio 84–98%" for re-rankers — **UNVERIFIED** (not on abstract page) — [arXiv HTML](https://arxiv.org/html/2604.14227)

#### VersionRAG / VersionQA
- "VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents", Daniel Huwiler, Kurt Stockinger, Jonathan Fürst; arXiv 2510.08109 (9 Oct 2025); VersionQA = 100 manually curated questions over 34 versioned technical documents; accuracy Naive RAG 58%, GraphRAG 64%, VersionRAG 90%; implicit change detection VersionRAG 60% vs baselines 0–10%; 97% fewer indexing tokens than GraphRAG; no GitHub on abstract page — [arXiv abs](https://arxiv.org/abs/2510.08109)

#### EnterpriseRAG-Bench
- "EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge", Yuhong Sun, Joachim Rahmfeld, Chris Weaver, Weijia Chen, Roshan Desai, Wenxi Huang (Onyx), Mark H. Butler (UC Berkeley); arXiv 2605.05253 v2 (19 May 2026) — [arXiv HTML](https://arxiv.org/html/2605.05253)
- Corpus 511,962 synthetic documents across 9 source types; 500 questions in 10 categories (+100 metadata-dependent extras); "Conflicting Info" = 20 questions, always 2 gold docs; built as near-duplicate pairs where a newer version at a new location changes specific facts, question generator told which supersedes; gold = up-to-date answer with anti-hallucination facts catching outdated-only responses — [arXiv HTML](https://arxiv.org/html/2605.05253)
- Metrics: Correctness (binary LLM judge), Completeness, Document Recall@10, Invalid Extra Documents, leaderboard score; Conflicting-Info correctness/recall: BM25 90.0/82.5, Vector (text-embedding-3-large) 60.0/55.0, Bash Agent (GPT-5.4 low) 80.0/72.5 — [arXiv HTML](https://arxiv.org/html/2605.05253)
- Code [github.com/onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench); leaderboard [HF space](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard); license not stated — [arXiv HTML](https://arxiv.org/html/2605.05253)

#### SituatedQA, FreshQA, TAQA ("Set the Clock"), CRAG — temporal context sets (not inter-document conflict per se)
- SituatedQA (Michael J.Q. Zhang, Eunsol Choi; EMNLP 2021; arXiv 2109.06157): NQ-Open questions whose answer depends on time or place; ~6.7K temporal, ~5.5K geographic; ~16.5% of NQ-Open context-dependent; retrieval with updated corpus ~15 points less accurate on changed answers; data at situatedqa.github.io — [arXiv abs](https://arxiv.org/abs/2109.06157); [ACL Anthology](https://aclanthology.org/2021.emnlp-main.586). SituatedQA Geo (105) and Temp (41) feed CONFLICTS — [arXiv 2506.08500v2](https://arxiv.org/html/2506.08500v2)
- FreshQA (FreshLLMs; Tu Vu et al.; Findings ACL 2024; arXiv 2310.03214): 600 questions (never-/slow-/fast-changing, false-premise); 500 test / 100 dev; periodically updated at [github.com/freshllms/freshqa](https://github.com/freshllms/freshqa) — [arXiv abs](https://arxiv.org/abs/2310.03214); 95 FreshQA queries feed CONFLICTS — [arXiv 2506.08500v2](https://arxiv.org/html/2506.08500v2)
- "Set the Clock: Temporal Alignment of Pretrained Language Models", Bowen Zhao, Zander Brumbaugh, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith; arXiv 2402.16797 (v2 9 Jun 2024); TemporalAlignmentQA (TAQA) 20,148 questions; aligning to 2010 gives 2.8× gain; code [github.com/yizhongw/llm-temporal-alignment](https://www.github.com/yizhongw/llm-temporal-alignment) — [arXiv PDF](https://arxiv.org/pdf/2402.16797). This is a parametric temporal-alignment set, not a retrieved-document conflict set.
- CRAG (Meta, KDD Cup 2024; arXiv 2406.04744): 4,409 QA pairs, 5 domains, mock web/KG APIs, dynamism from seconds to years; LLMs ~34% accuracy, straightforward RAG ~44% — [arXiv PDF](https://arxiv.org/pdf/2406.04744). "Corrective RAG" (CRAG, Yan et al. 2024) is a method, not a conflict dataset — **not investigated further**.

### Inferences
- HoH is the only *large* public temporal set with explicit per-version timestamps on both outdated and current evidence, so it can be re-packaged to score "selected the current document" directly; its ground truth assumes newest-is-correct, which matches a recency-based authority heuristic but cannot reward `unresolved_conflict`.
- Stale-Document Poisoning is the only dataset whose domains (law, platform policy, medicine, software) and "dated official sources" match the enterprise/regulatory supersession use case, but as of the abstract page no data download exists — treat as "contact authors".
- EnterpriseRAG-Bench's Conflicting-Info is tiny (20) but is the only enterprise-style superseding-document test with public code; BM25 at 90% correctness suggests it is not yet hard.

### Gaps
- HoH paper-level metrics and headline numbers not extracted (only qualitative abstract).
- Whether Re²Bench data is downloadable from the Re3 repo not verified; Re²Bench instances are retrieval-side (R@5/MRR) with generation accuracy, no per-document "obsolete" label field confirmed.
- VersionRAG/VersionQA public release not confirmed.
- Stale-Document Poisoning: no code/data link found.

---

## KQ5. Abstention-oriented RAG benchmarks and 2026 conflict-aware methods (do they add datasets?)

### Takeaway
2026 abstention benchmarks (LIT-RAGBench, GRAB-RAG, RINSE paired set, DeclineCraft) score `insufficient_evidence`-style refusals, but only LIT-RAGBench has a (3-question) contradictory-evidence slice; the 2026 conflict-aware method papers (ConflictRAG, KC-Bench, SIGIR-2026 ConflictQA) either release no new QA dataset or target text-vs-KG / agentic settings.

### Cited Findings
- LIT-RAGBench (Koki Itai et al., neoAI/Tokyo; arXiv 2603.06198, 6 Mar 2026): 114 Japanese questions + English version; Abstention category has Insufficient Evidence, Contradictory Evidence (3 questions, supporting evidence manually edited to conflict; expected: abstain or explicitly flag the contradiction), Incomplete Chunk; GPT-4.1 judge, binary accuracy, Over-Abstention Rate; Claude-Sonnet-4 "detected inconsistencies in all tasks and correctly abstained", Abstention category 0.950 (JA) / 0.967 (EN); overall best GPT-5 0.872; code [github.com/Koki-Itai/LIT-RAGBench](https://github.com/Koki-Itai/LIT-RAGBench) — [arXiv HTML](https://arxiv.org/html/2603.06198v1)
- GRAB-RAG ("Prompt-Based Abstention Fails Under Misleading Context", Yohanes Andre Setiawan; arXiv 2608.22228, 23 Aug 2026; CC BY 4.0): NQ + HotpotQA under supportive / degraded / missing / misleading contexts (gold passage edited to support a wrong answer); models still answered 41.6% of misleading questions under abstention prompting, 63% echoing the planted entity; generator-side conflict check cut to 13.3%; code/data "to be released" — [arXiv abs](https://arxiv.org/abs/2608.22228)
- RINSE ("Relevance Is Not Sufficient Evidence", Suting Chen, Peichun Hua, Yunming Xiao; arXiv 2609.37469, 27 Sep 2026): paired insufficient-evidence benchmark across six datasets (substitution / deletion / question-swap); 12 generators answer 40.0–99.3% of insufficient-evidence questions when told to abstain; RINSE 0.837 vs best prior 0.746; no repo listed — [arXiv abs](https://arxiv.org/abs/2609.37469)
- DeclineCraft (arXiv 2610.28340): 700 answerable + 300 unanswerable questions from Minecraft update history, Correct Abstention Rate across 27 LLMs — from search snippet only, **UNVERIFIED** — [arXiv abs](https://arxiv.org/abs/2610.28340)
- ConflictRAG ("ConflictRAG: Detecting and Resolving Knowledge Conflicts in Retrieval Augmented Generation", Chenyu Wang, Yueyuan Li, Yingmin Liu, Yang Shu; arXiv 2605.17301, v2 8 Jun 2026): two-stage detection (MLP then LLM) 90.8% detection accuracy at 62% lower API cost, conflict-detection F1 88.7%, entropy-TOPSIS source-credibility scoring +7.1% selection accuracy, "Conflict-Aware RAG Score (CARS)"; evaluated on "three benchmarks" (unnamed on abstract page); no dataset release or repo listed — [arXiv abs](https://arxiv.org/abs/2605.17301)
- ConflictQA (SIGIR 2026; "Exploring Knowledge Conflicts for Faithful LLM Reasoning: Benchmark and Method", Tianzhe Zhao, Jiaoyan Chen, Shuxiu Zhang, Haiping Zhu, Qika Lin, Jun Liu; arXiv 2604.11209, 13 Apr 2026): text-vs-knowledge-graph conflicts; 802 Non-Complementary + 430 Complementary samples (WebQSP, CWQ); TripleConf (KG misleading) vs TextConf (text misleading); metrics macro-F1 / EM; e.g. Non-COMP TripleConf best F1 54.63 (DeepSeek-V3.2-Thinking), TextConf best F1 76.24 (Qwen3-8B); code [github.com/Tianzhe26/ConflictQA](https://github.com/Tianzhe26/ConflictQA); license not stated — [arXiv HTML v1](https://arxiv.org/html/2604.11209v1). Note the name collides with Xie et al.'s ConflictQA (KQ3).
- KC-Bench ("KC-Bench: A Dynamic Interactive Benchmark for Evaluating Knowledge Conflicts in LLM Agents", Yaxing Lyu, Shengjie Zhou, Binbin Toh, Pengyu Zhu, Lijun Li; arXiv 2609.03588, 3 Sep 2026): 238 multi-turn agent tasks (screened from >1,000) covering world-knowledge conflicts, input inconsistencies, multi-source temporal conflicts; 9 models (DeepSeek-V4-Flash, GLM-5.2, MiniMax-M3 …); no repo listed — [arXiv abs](https://arxiv.org/abs/2609.03588)
- Search snippet also surfaced "From Facts to Conclusions: Integrating Deductive Reasoning in Retrieval-Augmented LLMs" (arXiv 2512.16795) as building on CONFLICTS — not opened, **UNVERIFIED** — [arXiv HTML](https://arxiv.org/html/2512.16795v1)

### Inferences
- None of the 2026 abstention sets labels *why* abstention is correct (missing vs contradictory) at scale except Ragability (KQ1) and LIT-RAGBench's 3 items; a team wanting to separate `insufficient_evidence` from `unresolved_conflict` will need to compose FaithEval-Unanswerable (2.4K) + FaithEval-Inconsistent (1.5K), or Ragability's two refusal conditions.

### Gaps
- ConflictRAG's three evaluation benchmarks are unnamed on the abstract page; no code found.
- KC-Bench and RINSE metrics/data release not confirmed.

---

## KQ6. Legal / regulatory precedence, policy supersession and enterprise document-version QA

### Takeaway
No public dataset was found that scores "which version of a law/policy is in force" or document status (active/draft/superseded); the closest are Stale-Document Poisoning (law & platform-policy reversals, no release found), EnterpriseRAG-Bench Conflicting-Info (20 synthetic superseding pairs) and VersionQA (34 versioned technical docs). Legal RAG sets found are retrieval/grounding benchmarks without version conflicts.

### Cited Findings
- Legal/regulatory QA sets surfaced by search (none tests version precedence): ObliQA-MP multi-passage regulatory QA ([NLLP 2025 PDF](https://aclanthology.org/2025.nllp-1.10.pdf)); Bar Exam QA and Housing Statute QA, "A Reasoning-Focused Legal Retrieval Benchmark" (Stanford/Princeton; [HF collection](https://huggingface.co/collections/reglab/a-reasoning-focused-legal-retrieval-benchmark); [ACM](https://dl.acm.org/doi/pdf/10.1145/3709025.3712219)); ClaimRAG-LAW, 317 QA pairs + 968 claims from GDPR / French Civil Code ([arXiv 2605.21071](https://arxiv.org/html/2605.21071)); Isaacus legal-rag-qa, 138 items ([HF](https://huggingface.co/datasets/isaacus/legal-rag-qa)); "Citation-Closure Retrieval and Per-Rule Attribution for Real-World Regulatory Compliance QA" ([arXiv 2605.29742](https://arxiv.org/pdf/2605.29742)). All from search snippets only; details **UNVERIFIED**.
- Search for enterprise version-conflict benchmarks returned EnterpriseRAG-Bench (KQ4) plus several practitioner blogs; blog references to "ERAA-2026" and "TemporalRAG-Bench 2026 (Stanford/NIST)" had no primary source and should be treated as **UNVERIFIED / possibly non-existent** — [ragaboutit blog](https://ragaboutit.com/5-temporal-rag-failures-85-of-enterprises-miss/)

### Inferences
- A pipeline that keys on status/scope/recency metadata will have to be evaluated on a custom-built or adapted set; the most reusable scaffolds are HoH (dated versions), CONFLICTS-Outdated (62 real queries with per-result `date`), EnterpriseRAG-Bench's generation pipeline (open code for superseding near-duplicates), and QACC's annotator "Time"/"Source" reasons.

### Gaps
- No dataset with explicit `status` ∈ {active, draft, superseded} fields was found anywhere in the 2023–2026 literature searched.
- Stale-Document Poisoning data availability unknown.

---

## KQ7. Cross-cutting: which datasets support (a) answer accuracy under conflict, (b) per-document source selection, (c) abstain / unresolved as correct

### Takeaway
(a) is supported by every set; (b) by RAMDocs, MacNoise, RAGuard, CONFACT, HoH, QACC, ConflictBank, ClashEval, ConflictingQA, WhoQA, WikiContradict (per-passage answers), MAGIC (sentence localisation); (c) by WikiContradict, FaithEval-Inconsistent, CONFLICTS (type-dependent), Ragability, LIT-RAGBench, and indirectly WhoQA/AmbigDocs ("No Answer"/clarification counted as a distinct failure type rather than correct).

### Cited Findings
- RAMDocs correctness requires all gold answers and no misinformation-only answers; documents carry `correct`/`misinfo`/`noise` — [arXiv HTML](https://arxiv.org/html/2504.13079); [GitHub](https://github.com/HanNight/RAMDocs)
- WikiContradict T5 "Correct" requires naming all contradictory answers without preference and flagging the contradiction; merely listing both without noting the contradiction is Incorrect — [arXiv HTML](https://arxiv.org/html/2406.13805)
- FaithEval-Inconsistent gold answer is "conflict" — [GitHub](https://github.com/SalesforceAIResearch/FaithEval)
- CONFLICTS: Conflicting opinions → neutral summary; Outdated → prioritise up-to-date; Misinformation → disregard; `correct_answer` only for specific types — [arXiv HTML v2](https://arxiv.org/html/2506.08500v2); [GitHub](https://github.com/google-research-datasets/rag_conflicts)
- Ragability `refusal_not_answerable` rewards refusal when context is missing *or* contradictory; `correct_answer_all` counts refusal, correct answer, or conflict identification as correct depending on condition — [LREC 2026 PDF](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.182.pdf)
- WhoQA counts a response correct only if all supported answers are included; gpt-3.5-turbo's clarification requests on 99% of hard questions are reported as rejections, not correct — [arXiv HTML](https://arxiv.org/html/2410.15737)
- AmbigDocs "No Answer" (abstention) is one of the four *incomplete* categories — [arXiv HTML v3](https://arxiv.org/html/2404.12447v3)
- QACC annotators record which contexts support which answer (attribution) and the reason the correct answer wins (Majority / Source / Common Sense / Time) — [arXiv HTML v3](https://arxiv.org/html/2410.12311v3)
- MacNoise/entity-replacement passages carry binary perturbed labels and discriminator F1 is reported (81.00 / 80.62 at 35%) — [arXiv HTML v3](https://arxiv.org/html/2305.01579v3)
- MAGIC scores localisation of conflicting sentences (LOC) — [arXiv HTML v3](https://arxiv.org/html/2507.21544v3)

### Inferences
- Mapping of the pipeline's four decisions to existing gold behaviours: `answered` — all sets; `unresolved_conflict` — WikiContradict T5, FaithEval-Inconsistent, CONFLICTS "Conflicting opinions", Ragability contradictory condition, LIT-RAGBench Contradictory Evidence; `needs_information` — no dataset scores a clarification request as correct (WhoQA/AmbigDocs penalise it; SituatedQA/CONFLICTS-Complementary are the nearest conceptual matches); `insufficient_evidence` — FaithEval-Unanswerable, Ragability missing-context condition, LIT-RAGBench Insufficient Evidence, GRAB-RAG missing, RINSE paired set, DeclineCraft.
- For scoring "which document was selected", RAMDocs (`type`), HoH (`outdated_infos` vs current `evidence`), CONFACT (credibility/stance), RAGuard (supporting/misleading/unrelated) and QACC (attribution) can be used without re-annotation; CONFLICTS and WikiContradict would need a per-document "authoritative" label added.

### Gaps
- No benchmark found that treats a clarification request (`needs_information`) as the gold response in a multi-document conflict setting.
- No benchmark found that labels, per instance, whether the conflict is *resolvable by authority* vs *genuinely unresolved*, except CONFLICTS' query-level taxonomy (Outdated/Misinformation = resolvable; Conflicting opinions = unresolved).
