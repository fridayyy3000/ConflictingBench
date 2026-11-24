# GOV-RAG vs baseline on external benchmarks (Nemotron 3.5 Super-VL 120B)

Generated 2026-11-20 from the four full runs in this directory. Generator: `nvidia/nemotron-3.5-super-vl-120b-a12b`; judge (CONFLICTS only): `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`; embedder: `BAAI/bge-small-en-v1.5` on CPU; retrieval top-30 -> top-8 in both modes. Baseline = same retrieval, semantic top-8, passages-only prompt, same generator and validator.

## CONFLICTS (458 questions)

| Metric | Baseline | GOV-RAG | Delta |
|---|---|---|---|
| Decision match % | 59.83 | 66.59 | +6.76 |
| Answer correct % (n=237 with reference) | 65.4 | 71.73 | +6.33 |
| Judge adherence % | 64.19 | 74.24 | +10.05 |
| Mean latency s | 40.66 | 40.2 | -0.46 |
| Decision match 95% CI | [55.7, 64.2] | [62.2, 71.0] | |
| Judge adherence 95% CI | [59.6, 68.6] | [70.3, 78.2] | |
| Final validation failures (after repair/salvage) | 88 | 39 | |

Paired (same question): decision match GOV-RAG right/baseline wrong = 85, reverse = 54. Judge adherence GOV-RAG only = 87, baseline only = 41. Answer correct GOV-RAG only = 29, baseline only = 14.

### By conflict type (decision match % / judge adherence %)

| Type | n | Baseline | GOV-RAG |
|---|---|---|---|
| Complementary information | 115 | 76.52 / 85.22 | 60.0 / 95.65 |
| Conflict due to misinformation | 5 | 80.0 / 80.0 | 60.0 / 60.0 |
| Conflict due to outdated information | 62 | 80.65 / 56.45 | 72.58 / 69.35 |
| Conflicting opinions and research outcomes | 115 | 10.43 / 46.96 | 46.96 / 71.3 |
| No conflict | 161 | 74.53 / 63.98 | 83.23 / 63.35 |

### Decision confusion (rows = expected, cols = predicted)

**Baseline**

| expected_decision   |   answered |   insufficient_evidence |   needs_information |   unresolved_conflict |
|:--------------------|-----------:|------------------------:|--------------------:|----------------------:|
| answered            |        262 |                      77 |                   1 |                     3 |
| unresolved_conflict |         82 |                      21 |                   0 |                    12 |

**GOV-RAG**

| expected_decision   |   answered |   insufficient_evidence |   needs_information |   unresolved_conflict |
|:--------------------|-----------:|------------------------:|--------------------:|----------------------:|
| answered            |        251 |                      42 |                  27 |                    23 |
| unresolved_conflict |         52 |                       9 |                   0 |                    54 |

Reading: GOV-RAG recognises genuine opinion/research conflicts far more often (54 vs 12 of 115 expected `unresolved_conflict`), which drives the decision-match and adherence gains. Its cost is over-caution on answerable items: it routes 92 expected-`answered` items to a non-answer decision versus 81 for the baseline, mostly `needs_information` on complementary-information items. Complementary information is the one type where GOV-RAG's decision match is lower (60.0 vs 76.5) even though its judge adherence there is the highest of any cell (95.7).

## RAMDocs (500 questions, set-EM protocol from MADAM-RAG)

| Metric | Baseline | GOV-RAG | Delta |
|---|---|---|---|
| Set-EM % | 41.4 | 38.2 | -3.2 |
| Strict set-EM % (gold verbatim in prose) | 37.8 | 36.6 | -1.2 |
| Any gold in answer % | 91.4 | 93.2 | +1.8 |
| Misinfo value in answer set % | 26.0 | 35.0 | +9.0 |
| Cited-doc precision % | 80.8 | 78.55 | -2.25 |
| Gold doc in context % | 99.4 | 99.4 | +0.0 |
| Mean latency s | 31.12 | 32.65 | +1.53 |
| Set-EM 95% CI | [37.0, 45.4] | [34.0, 42.4] | |
| Set-EM by #gold (1/2/3) | {'1': 60.0, '2': 44.0, '3': 29.5} | {'1': 49.0, '2': 42.0, '3': 29.0} | |
| Decisions | {'answered': 461, 'unresolved_conflict': 24, 'insufficient_evidence': 15} | {'answered': 401, 'unresolved_conflict': 85, 'insufficient_evidence': 14} | |
| Selected source label | {'correct': 424, 'none': 39, 'noise': 17, 'misinfo': 20} | {'correct': 372, 'none': 99, 'noise': 16, 'misinfo': 13} | |
| Final validation failures | 5 | 0 | |

Paired set-EM: GOV-RAG right/baseline wrong = 22, reverse = 38, both right = 169, both wrong = 271.

### Set-EM by decision

| Decision | Baseline n / set-EM / misinfo-in-set | GOV-RAG n / set-EM / misinfo-in-set |
|---|---|---|
| answered | 461 / 44.9 / 23.0 | 401 / 47.6 / 23.7 |
| unresolved_conflict | 24 / 0.0 / 100.0 | 85 / 0.0 / 94.1 |
| insufficient_evidence | 15 / 0.0 / 0.0 | 14 / 0.0 / 0.0 |

Reading: on the items both systems answer, GOV-RAG is slightly better (47.6% vs 44.9% set-EM) and it selects a misinformation source less often (13 vs 20). The overall deficit comes entirely from the 85 items GOV-RAG labels `unresolved_conflict` (baseline: 24). In those, the model lists every competing value, including the planted misinformation value, in `short_answers`; set-EM then scores the item wrong by construction (0/85 correct, 94% contain a misinfo value) even though 95% of them contain the gold answer. Of the 61 items where GOV-RAG said conflict and the baseline committed to an answer, the baseline was right on 28%. RAMDocs web snippets carry no authority, date or scope signals, so the governance reranker has nothing to govern with and the system falls back to surfacing the disagreement, which is what GOV-RAG is designed to do but is exactly what set-EM penalises.

## Implications for the paper

- CONFLICTS is the benchmark that matches GOV-RAG's task: +6.8 decision match, +6.3 answer correctness, +10.1 judge adherence, all with non-overlapping or barely overlapping 95% CIs for adherence. Report it as the primary external result.
- RAMDocs should be reported honestly as roughly tied on set-EM (38.2 vs 41.4, overlapping CIs) with the explanation above, plus a conflict-aware scoring variant (e.g. count an `unresolved_conflict` whose listed set contains all gold values as 'surfaced') as a secondary column. Under that variant GOV-RAG would gain up to 81 items; compute it before claiming a number.
- Two cheap fixes worth an ablation: (1) when no governance signals are present for any candidate, fall back to majority/consistency voting instead of declaring conflict; (2) exclude values the model itself labels as minority or contradicted from `short_answers`.
- Over-caution on CONFLICTS complementary-information items (`needs_information` 19 times) is the other ablation target; the baseline never produces it.
