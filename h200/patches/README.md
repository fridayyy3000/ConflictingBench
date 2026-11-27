# Final answer-and-flag policy (A′ + B + E + E′): code and outcome, for the record

> **Update, 27 Nov 2026 (supersedes the outcome below).** The user overrode the stop rule on the
> evening of 26 Nov and the six full runs of `answer_and_flag_final` ran unchanged (completed
> 27 Nov 00:23; tag `*_govrag_hybrid_answerandflag_final`; gpt-oss-120b judge). **User decision,
> 27 Nov: the final policy is the paper's GOV-RAG system row.** Variant 3
> (`govrag_hybrid_answerandflag`) is the ablation "answer-and-flag without commitment"; A-only stays
> an ablation row. `build_judge_table.py --paper` uses the final row. The pre-check 2 failure below
> stands as recorded. On this branch the patches are applied to `corvic/` (`pipeline_final.patch`
> as is; `runner_final_wiring.patch` ported by hand, without the H200-only `--provider local`,
> `--run-label` and `--base-url` changes it did not contain either).

Written on the H200, 26 Nov 2026, following `h200/BRIEF_final_policy.md` (commits e3edffb and
f90cac9, including Revision 1). **Outcome: pre-check 2 failed set 1, so under the stop rule the six
full runs were not started. The paper keeps the current table (system row = Variant 3,
`govrag_hybrid_answerandflag`).** No baseline, strict, dataset, decision label, judge prompt or
temperature was changed.

## Files

| file | what |
|---|---|
| `pipeline_final.patch` | `diff -u` of `corvic/pipeline/gov_rag_gemini.py` (branch original, identical to the H200 copy before 26 Nov) against the H200 version. Applies cleanly to the Mac file. Besides the final policy it also contains the earlier H200 policies tried that day, each behind its own `--govrag-policy` value: `two_stage`, `answer_and_flag_v2` (A + C), `answer_and_flag_a` (A). |
| `runner_final_wiring.patch` | the final-policy wiring only, in `pipeline/providers.py` and `evaluate_external.py`: policy routing, `enforce_committed`, the `_final` tag suffix, `--qids-file`, and the `conflict_free` / `answer_rebuilt` columns. Diffed against the H200 copies from just before the final-policy work, because those files carry H200-only local-vLLM changes. |
| `answer_and_flag_final_prompt.txt` | the final system prompt, verbatim. |

## What each change does

- **A** (`answer_and_flag_a`): corroboration decides factual claims only. Opinions, recommendations
  and research outcomes that governance does not separate are `unresolved_conflict`, with each
  position laid out neutrally in the answer, sources named, and each position listed in
  `short_answers` (A's text kept exactly as validated).
- **A′** (Revision 1): `unresolved_conflict` is evaluative-only, and a factual disagreement is never
  one. The tie rule becomes "factual disagreements are always resolved": corroboration, then
  consistency, then the more specific, internally consistent passage with confidence "low". Never
  from background knowledge. The brief's text numbers this "step 4" against the base prompt; in the
  A prompt it is step 6, with corroboration and consistency as steps 4 and 5.
- **B**: one prompt sentence ("When the passages do not disagree and at least one of them addresses
  the question, answer it; …") plus a retrieval branch in `GovRAGGemini.query`. When the claim-bearing
  candidates carry at most one distinct normalized claim, rank by retrieval score as the baseline
  does, with no claim diversification. Salvage after the single repair was already the default in
  `run_generation`. **Finding:** the conflict-free test is true on only 18/500 RAMDocs and 2/458
  CONFLICTS items, because `extract_claim` returns any number in a chunk and 30 retrieved chunks
  almost always carry several. B works almost only through its prompt sentence.
- **E**: the `answer` field states committed values only; a rejected value may appear only in a
  final sentence that marks it as rejected. Schema lines for `answer` and `explanation` as in the
  brief. E's paragraph keeps "when you commit under step 4 or 5", so A′'s step-6 commit is covered
  by E′, not by that sentence.
- **E′** (Revision 1): a validator check for `answered` results, active for this policy only
  (`run_generation(..., enforce_committed=True)`). A violation goes through the existing single
  repair; if the repair still names the value outside a sentence containing "reject" or
  "contradict", the answer is rebuilt from `short_answers`, one sentence per committed value, and
  recorded as `answer_rebuilt`. **Deviation from the brief, decided by the user before pre-check 2:**
  - Rejected values are the model's own new `rejected_values` schema field, plus the numeric claim
    of each `conflicting` citation unless a governing or supporting cited chunk carries the same
    claim.
  - The brief's corroboration-summary values were dropped. On Gemma's Variant 3 RAMDocs answers
    they flagged 140 answers for incidental numbers (mostly years), against 11 for planted values.
  - The numeric extractor alone could see the planted value on 42 of the 127 hedging items at best,
    and on none of the 52 with name-valued planted answers.

## Numbers from the brief (current system, `govrag_hybrid_answerandflag`, gpt-oss-120b judge)

| generator | RAMDocs PC for hedging | RAMDocs Inc for refusing | CONFLICTS Inc for refusing |
|---|---|---|---|
| Nemotron | 60 / 500 (12.0 pts) | 9 (1.8) | 16 (3.5) |
| Qwen | 42 / 500 (8.4 pts) | 17 (3.4) | 22 (4.8) |
| Gemma | 127 / 500 (25.4 pts) | 17 (3.4) | 23 (5.0) |

## Pre-check results (Gemma, hybrid, one pass each, gpt-oss-120b judge)

Set 1: the 127 Gemma RAMDocs hedging items.

| run | FC / PC / Inc | answered / unresolved |
|---|---|---|
| Variant 3 (current) | 0 / 127 / 0 | 103 / 24 |
| pre-check 1: A + B + E | 13 / 107 / 7 | 53 / 74 |
| pre-check 2: A′ + B + E + E′ | 51 / 53 / 23 | 114 / 13 |

| pre-check 2 condition | result | required | |
|---|---|---|---|
| FC | 51 / 127 | ≥ 50 | pass |
| `short_answers` unchanged, of the 103 Variant 3 answered | 52 | ≥ 85 | **fail** |
| answer names a `wrong_answers` value without a rejecting sentence | 56 | ≤ 15 | **fail** |
| `answer_rebuilt` | 2 | ≤ 30 | pass |

Set 2: 40 CONFLICTS opinion items (the first 40 qids where Gemma A-only was FC). A-only: 40 FC
(39 unresolved). Pre-check 2: 39 FC, 1 PC (38 unresolved). Unresolved and FC: **37 ≥ 32, pass.**

Pre-check 1 also failed all three of its conditions (FC 13 ≥ 60; wrong value named 109 ≤ 25;
`short_answers` unchanged 57.5% ≥ 90%).

Why set 1 still fails:
- A′ fixed the routing: unresolved fell from 74 to 13. But Gemma now often puts the planted value
  into `short_answers` itself, e.g. q0006 `1902 | 1885 | July 10, 1850`, or q0020
  `university | academy/high school`.
- A value in `short_answers` counts as committed, so E′ (which never looks at gold) cannot treat it
  as rejected. That is why only 2 answers were rebuilt while 56 still name a wrong value.
- Inc rose from 0 to 23 on these items.

Outputs: `results_precheck/` (pre-check 1) and `results_precheck2/` (pre-check 2), with details,
metrics, judge files and the qid lists. `tools/precheck_final.py` and `tools/precheck2_final.py`
reprint the reports.
