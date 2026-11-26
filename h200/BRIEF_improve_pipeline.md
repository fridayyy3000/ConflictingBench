# Brief for the H200 session: close the CONFLICTS gap without breaking RAMDocs

You are working in `/mnt/data1/gnajpand/conflictbench_h200`. The pipeline under test is
`corvic/pipeline/gov_rag_gemini.py` (class `GovRAGGemini`, method `query`, line ~1782). The
runner is `corvic/evaluate_external.py` with `--govrag-retrieval hybrid --govrag-policy
answer_and_flag` for the paper system. Judging is `corvic/judge_three_way.py` with judge
`openai/gpt-oss-120b`, via `h200/judge_all.sh`. Do not touch anything under `data/external/`,
do not print or copy the contents of `corvic/.env`, keep temperature 0 everywhere, do not
change the judge prompt or the baselines.

## The problem

With the gpt-oss-120b judge (FC / Inc), the system is best or tied on RAMDocs for Nemotron and
Qwen but never leads on CONFLICTS:

| CONFLICTS, FC / Inc | Nemotron | Qwen3.8-27B | Gemma-4-31B |
|---|---|---|---|
| Hybrid RAG | 62.9 / 28.8 | 55.9 / 35.6 | 70.1 / 17.5 |
| GOV-RAG strict + hybrid | 70.3 / 22.9 | 63.8 / 27.1 | 67.2 / 20.3 |
| System: answer-and-flag + hybrid | 64.6 / 28.6 | 57.9 / 33.2 | 67.9 / 19.9 |

## Where the points go (computed on the Mac from your judge files, 26 Nov)

FC / Inc per conflict type, gpt-oss-120b judge.

Nemotron:

| Conflict type (n) | Hybrid RAG | LlamaIndex k8 | Rankify | strict+hybrid | System |
|---|---|---|---|---|---|
| No conflict (161) | 80.1 / 10.6 | 82.6 / 10.6 | 83.2 / 9.9 | 72.0 / 14.3 | 74.5 / 11.2 |
| Complementary (115) | 75.7 / 8.7 | 66.1 / 20.9 | 61.7 / 20.0 | 87.8 / 7.0 | 91.3 / 5.2 |
| Conflicting opinions (115) | 19.1 / 77.4 | 29.6 / 64.3 | 13.9 / 80.9 | 53.9 / 44.3 | 17.4 / 80.0 |
| Outdated (62) | 72.6 / 25.8 | 74.2 / 22.6 | 80.6 / 19.4 | 66.1 / 32.3 | 74.2 / 24.2 |
| Misinformation (5) | 100 / 0 | 80 / 20 | 100 / 0 | 40 / 60 | 100 / 0 |

Qwen3.8-27B:

| Conflict type (n) | Hybrid RAG | LlamaIndex k8 | Rankify | strict+hybrid | System |
|---|---|---|---|---|---|
| No conflict (161) | 78.3 / 14.3 | 73.9 / 10.6 | 80.1 / 11.2 | 70.2 / 17.4 | 70.8 / 15.5 |
| Complementary (115) | 51.3 / 31.3 | 78.3 / 14.8 | 53.9 / 28.7 | 76.5 / 13.9 | 72.2 / 15.7 |
| Conflicting opinions (115) | 18.3 / 77.4 | 19.1 / 70.4 | 5.2 / 90.4 | 39.1 / 54.8 | 16.5 / 79.1 |
| Outdated (62) | 72.6 / 24.2 | 72.6 / 24.2 | 80.6 / 17.7 | 67.7 / 27.4 | 71.0 / 29.0 |

Gemma-4-31B:

| Conflict type (n) | Hybrid RAG | LlamaIndex k8 | Rankify | strict+hybrid | System |
|---|---|---|---|---|---|
| No conflict (161) | 65.2 / 16.8 | 67.7 / 16.1 | 72.0 / 13.7 | 58.4 / 22.4 | 64.0 / 18.0 |
| Complementary (115) | 67.0 / 18.3 | 68.7 / 15.7 | 75.7 / 14.8 | 67.8 / 17.4 | 69.6 / 16.5 |
| Conflicting opinions (115) | 78.3 / 17.4 | 72.2 / 27.8 | 64.3 / 30.4 | 80.0 / 20.0 | 70.4 / 23.5 |
| Outdated (62) | 72.6 / 19.4 | 71.0 / 19.4 | 82.3 / 14.5 | 67.7 / 19.4 | 67.7 / 25.8 |

What this says:

1. **Conflicting opinions is the whole gap.** On Nemotron the system is Inc on 92 of 115
   opinion items, every one of them `answered`. The judge's rationale is the same each time:
   "commits to a single view rather than neutrally presenting the conflicting opinions as
   required". Strict shows the fix already works: its 55 `unresolved_conflict` decisions on
   these items are FC 55 of 55, while its 60 `answered` decisions are Inc 51 of 60. Gemma
   hedges on its own, which is why every method scores 64 to 80 there. Lifting the system's
   opinion row to strict's level is worth roughly 8 to 10 points of overall CONFLICTS FC for
   Nemotron and Qwen, which puts the system ahead of every row in the main table.
2. **No conflict costs about 5 to 8 points on that slice**, almost entirely from refusals:
   the system returns `insufficient_evidence` on 11 to 13 of the 161 no-conflict items (plus a
   few `needs_information` for Qwen), and a refusal on a no-conflict question is never FC.
   Answered-but-wrong counts are close to hybrid RAG's.
3. **Outdated information is not a loss under this judge.** The system is within two points
   of hybrid RAG on Nemotron and Qwen. Do not change the recency logic.
4. The system already wins clearly on Complementary information for Nemotron (91.3 vs 75.7).
   Keep that behaviour intact: whatever you change must not reduce `answered` on items where
   the passages do not disagree.

## Step 1: confirm on your side before changing code

Join each judge checkpoint (`results_external/conflicts_*_judge3_openai-gpt-oss-120b_checkpoint.jsonl`,
one JSON per line with `qid`, `label` in {FC, PC, Inc}, `reason`) with the details CSV
(`conflicts_*_details_*.csv`, columns `qid, conflict_type, decision, predicted_answer,
gold_answers, reason`) and reproduce the tables above. Then print, for the system on
Nemotron, the 11 no-conflict items with `decision != answered`: question, `predicted_answer`,
`reason`, and the hybrid RAG `predicted_answer` and judge label on the same qid. Report what
the refusals have in common (off-topic retrieval, a quote the validator rejected, a question
the model treated as unanswerable) before implementing change B.

## Step 2: two changes in `corvic/pipeline/gov_rag_gemini.py`

Keep the four decision labels (`DECISIONS`, line ~329) exactly as they are so the runner, the
judge and the UI keep working. Do not touch the strict policy, the baselines, or any threshold
constant. No sweeps against the test sets: each rule is fixed from the design and checked once
on a 40-item sample.

### A. Do not force a winner on evaluative disagreements (the main lever)

Edit resolution step 2 in `SYSTEM_PROMPT_ANSWER_AND_FLAG` (line ~112). Corroboration decides
only for factual claims: a number, date, name, place, event, or any question where one value is
simply wrong. When the competing passages are opinions, recommendations, interpretations, or
research outcomes that can legitimately differ (typical questions: "Does X help with Y?",
"Is X a solution to Y?", "Are X more effective than Y?"), and step 1 (governance) does not
separate them, the decision is `unresolved_conflict`. In that case the `answer` field must
present each position neutrally with its source and what it rests on, in two to four
sentences, without saying which one is right. The dissenting passages are cited with
treatment `conflicting` as now. `short_answers` stays empty for `unresolved_conflict`. Keep the
tie rule and keep everything about factual conflicts unchanged: RAMDocs conflicts are factual
(misinformation and ambiguity) and the system must keep committing there.

Also check what the details CSV carries as `predicted_answer` for `unresolved_conflict` rows
(`score_item` in `corvic/evaluate_external.py`, line ~319, uses `result.get("answer")`). In the
strict run it is non-empty and the judge reads it, so it should already work; confirm.

### B. Stop refusing on questions with no competing value

In `query`, after `build_candidates` and before diversification, detect whether the retrieved
evidence contains competing values: group the claim-bearing candidates with
`normalize_claim`/`group_by_claim`; at most one distinct normalized claim means conflict-free.
For conflict-free questions, rank by retrieval score exactly as the baseline branch does
(fused score in hybrid mode, cosine otherwise) instead of the authority-weighted `final_score`,
keep the top `FINAL_CONTEXT_K`, and generate with the answer-and-flag prompt as usual so the
output stays structured and cited. Add one sentence to the prompt for this case: when the
passages do not disagree and at least one of them addresses the question, answer; reserve
`insufficient_evidence` for passages that are off-topic. If Step 1 shows the refusals come from
the validator (quote not verbatim after the single repair), make salvage
(`drop_unverifiable=True` in `_validate_generation_result`, line ~583) the default after the
repair instead of failing the whole answer. Record `conflict_free: true/false` in the result
dict so it lands in the details CSV.

## Step 3: sample, then full rerun

1. Fixed 40-item sample with Nemotron: 20 CONFLICTS (8 no-conflict, 8 opinion, 4
   complementary) and 20 RAMDocs. Compare decisions and answers before and after on the same
   items. Required: opinion items move to `unresolved_conflict` with both positions in the
   answer; no-conflict refusals drop; RAMDocs `answered` rate and set-EM do not fall;
   complementary items stay `answered`. One pass only. If a change clearly hurts, drop it and
   say so.
2. Full rerun of the system only (`--govrag-retrieval hybrid --govrag-policy answer_and_flag`)
   on both datasets for all three generators under a new tag suffix,
   `govrag_hybrid_answerandflag_v2`, so the current results stay as the pre-fix ablation.
3. Judge the new runs with gpt-oss-120b, same settings as before, then produce the main
   FC / Inc table with the v2 row added and the per-conflict-type table for v1, v2, hybrid RAG
   and strict+hybrid, per generator.
4. Save the code change as `patches/pipeline_v2.patch` (`diff -u` of the original and changed
   `gov_rag_gemini.py`) with a short `patches/README.md` stating what A and B do and the
   Step 1 numbers behind them.

## What done looks like

A report with: the Step 1 refusal analysis; the 40-item before/after; the full v2 rows for
three generators on both datasets with the gpt-oss judge; the per-type tables; the patch file.
No change to baselines, strict, datasets, decision labels, judge prompts, or temperatures.
Nothing printed from `.env`.
