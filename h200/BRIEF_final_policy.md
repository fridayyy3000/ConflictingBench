# Brief: the final answer-and-flag policy (A + B + E), rerun on all three generators

Written on the Mac, 26 Nov 2026, from the judge files pulled today. Supersedes the Nemotron
A-only rerun: **if that rerun is still going, stop it**; the final policy replaces it.

## Decision

The paper's system row becomes a single final policy, `answer_and_flag_final`, made of three
changes applied to every generator identically:

- **A** (already implemented and validated by you as `answer_and_flag_a`): corroboration decides
  factual claims only; evaluative disagreements without a governance signal are
  `unresolved_conflict` with both positions in the answer.
- **B** (from `BRIEF_improve_pipeline.md`, section B; implement it if it is not in the tree):
  no refusal when the retrieved evidence has no competing value and a passage addresses the
  question; rank conflict-free questions by retrieval score; salvage after the single repair
  instead of failing the answer.
- **E** (new, the biggest lever, below): the committed answer and the dissent are separated.

The current `govrag_hybrid_answerandflag` runs stay in the table as the ablation "dissent in
the answer". Nothing else moves: baselines, strict, judge prompts, decoding, datasets.

## Why E: what the judge is penalising

The judge rubric says: presenting a known misinformation value "as an unresolved alternative
alongside the gold answer is PC". Our prompt asks for exactly that. The schema line reads
`"answer": "... mention flagged dissent briefly."`, and Gemma and Qwen obey it by listing the
planted value as another option ("Additionally, one source lists a total population of
12,345"). Nemotron phrases the same dissent as rejected ("contradicted by multiple documents")
and is graded FC on most of the same items.

Share of the full set graded PC where the judge's reason names a misinformation value or an
alternative, and share graded Inc for refusing, under the current system
(`govrag_hybrid_answerandflag`, gpt-oss-120b judge):

| generator | RAMDocs PC for hedging | RAMDocs Inc for refusing | CONFLICTS Inc for refusing |
|---|---|---|---|
| Nemotron | 60 / 500 (12.0 pts) | 9 (1.8) | 16 (3.5) |
| Qwen | 42 / 500 (8.4 pts) | 17 (3.4) | 22 (4.8) |
| Gemma | 127 / 500 (25.4 pts) | 17 (3.4) | 23 (5.0) |

On Gemma RAMDocs the whole gap to Rankify (37.4 vs 45.4 FC) sits in these items: on clean
questions Gemma under our policy is level with Rankify (65 vs 68 FC), on questions with planted
misinformation it is 20 vs 31, and its Inc there is the same as Rankify's (30 vs 32), so the
system is not safer on them, it is hedging. Retrieval is not the cause: every system has the
gold document in context on 99.4% of items.

Three of the 47 Gemma items where Rankify is FC and we are not (41 of the 47 are PC):

- q0127 (gold 81,615 | 963,022; planted 12,345): answer gives both gold values, then
  "Additionally, one source lists a total population of 12,345 for Katwa, though it does not
  specify the exact entity scope". Judge: PC, "introduces a known misinformation value as an
  alternative".
- q0102 (gold Dynasty | Jeremy Camp; planted Metallica): "There are multiple albums titled
  'Beyond Measure' by different artists: Jeremy Camp, Dynasty, and Metallica." Judge: PC.
- q0118 (gold Parsons | Gregory; planted Frank Lloyd Wright): correct for both buildings, then
  "(though one source conflictingly attributes it to Frank Lloyd Wright)". Judge: PC.

In each case `short_answers` is already right; only the `answer` text hedges.

## E. Separate the committed answer from the dissent

In `SYSTEM_PROMPT_ANSWER_AND_FLAG` (`gov_rag_gemini.py`, line ~112 on the Mac; your copy of
the `answer_and_flag_a` prompt), make these edits. The same edits apply to the A prompt text
you derived from it, so that `answer_and_flag_final` = A prompt + B sentence + E.

1. Replace the paragraph that begins `When you commit under step 2 or 3` with:

```
When you commit under step 2 or 3, the decision is answered, conflict_detected is true, the
dissenting passage is cited with treatment "conflicting", and the explanation names the
rejected value and why it lost. The "answer" field states the committed answer(s) only. Do not
list a rejected value in the answer as an additional figure, another possibility, or a
qualification in parentheses; a reader of the answer alone must see one value per entity. If
you mention a rejected value in the answer at all, do it in a separate final sentence that
says it is rejected and why, for example: "One source gives 12,345; that figure is rejected
because it is contradicted by the other documents." short_answers must contain only the
committed answer(s), never the rejected value. If the question is ambiguous because several
distinct entities share the same name, give the correct answer for each entity; that is not a
conflict, and each entity's value is committed in the same way.
```

2. Replace the schema line for `answer` with:

```
  "answer": "Readable committed answer with all important conditions and exceptions. No rejected values except in a final sentence that marks them as rejected.",
```

3. Keep the `explanation` field as the place where the resolution and the dissent are
   described: `"explanation": "Brief explanation of this decision, including how any conflict
   was resolved and which value was rejected."`

4. Add the B sentence after the decision list (if B is implemented in the prompt rather than
   only in retrieval):

```
When the passages do not disagree and at least one of them addresses the question, answer it;
insufficient_evidence is for passages that are off-topic, not for passages that are merely brief.
```

Do not change the resolution order, the decision labels, the tie rule, the treatments, the
validator or the repair prompt. Do not write a per-generator prompt: one prompt for all three.

## Check on the hedging items before the full runs

Run Gemma, RAMDocs, hybrid retrieval, `answer_and_flag_final`, on these 127 qids only
(the Gemma items currently graded PC for hedging), then judge them with gpt-oss-120b as usual:

```
q0002 q0004 q0005 q0006 q0008 q0020 q0021 q0022 q0027 q0032 q0037 q0038 q0039 q0041 q0045 q0046 q0048 q0049 q0050 q0051 q0053 q0055 q0061 q0063 q0068 q0071 q0073 q0074 q0077 q0081 q0088 q0090 q0094 q0097 q0102 q0118 q0127 q0130 q0135 q0137 q0139 q0140 q0144 q0146 q0149 q0155 q0157 q0158 q0169 q0172 q0177 q0182 q0189 q0190 q0194 q0199 q0204 q0211 q0221 q0223 q0225 q0229 q0234 q0241 q0249 q0253 q0255 q0258 q0261 q0263 q0278 q0282 q0285 q0287 q0289 q0291 q0295 q0296 q0297 q0298 q0300 q0314 q0318 q0320 q0322 q0323 q0325 q0328 q0336 q0338 q0342 q0350 q0351 q0357 q0358 q0360 q0364 q0369 q0382 q0393 q0405 q0406 q0407 q0414 q0418 q0424 q0431 q0442 q0443 q0446 q0448 q0450 q0451 q0454 q0456 q0460 q0464 q0465 q0467 q0468 q0469 q0477 q0479 q0485 q0491 q0492 q0495
```

Required, one pass, no tuning: the `answer` field names a value from `wrong_answers` in at
most 25 of them, and where it does, the sentence marks it as rejected; `short_answers` is
unchanged on at least 90% of them; FC on these items goes from 0 to at least 60 of 127.
If the FC gain is below that, send me five answers that are still PC with the judge reasons
before doing anything else. If the edits clearly hurt (committed values change on many
items), stop and say so.

The same qid lists for the other two generators, if you want a quick look (not required):

- Nemotron (60): `q0004 q0005 q0015 q0018 q0022 q0026 q0046 q0048 q0049 q0077 q0090 q0094 q0095 q0096 q0097 q0102 q0116 q0133 q0135 q0137 q0149 q0155 q0156 q0157 q0159 q0164 q0194 q0204 q0211 q0225 q0229 q0253 q0256 q0261 q0264 q0267 q0289 q0297 q0300 q0314 q0322 q0342 q0351 q0357 q0370 q0375 q0381 q0389 q0391 q0392 q0413 q0424 q0431 q0442 q0455 q0465 q0485 q0492 q0495 q0496`
- Qwen (42): `q0004 q0005 q0006 q0008 q0021 q0027 q0032 q0037 q0039 q0041 q0045 q0051 q0053 q0055 q0061 q0068 q0071 q0073 q0077 q0080 q0088 q0094 q0102 q0155 q0157 q0190 q0211 q0218 q0234 q0236 q0266 q0273 q0284 q0291 q0300 q0303 q0328 q0329 q0364 q0469 q0474 q0485`

## Full runs

Six runs under the tag `govrag_hybrid_answerandflag_final`, full sets (500 RAMDocs, 458
CONFLICTS), `--govrag-retrieval hybrid --govrag-policy answer_and_flag_final`:

| generator | thinking | where | note |
|---|---|---|---|
| nvidia/nemotron-3.5-super-vl-120b-a12b | **on** (as every other Nemotron row) | API, 3 workers | about 1.5 h per dataset |
| Qwen 3.8 27B | off (as every other Qwen row) | GPU 6, vLLM | |
| Gemma 4 31B | off (as every other Gemma row) | GPU 6, vLLM | |

Order: Gemma RAMDocs first (the cell we most need to see), then Gemma CONFLICTS, Qwen both,
Nemotron both on the API in parallel with the GPU runs. Record the thinking setting in the run
log and the metrics JSON for every run. Judge each run with gpt-oss-120b, same protocol and
temperature as before, one pass.

Then:

1. Rebuild your judge table with `govrag_hybrid_answerandflag_final` as the main row and the
   current `govrag_hybrid_answerandflag` as an ablation row, and send me the FC / Inc of the
   final row per cell next to the current system, Hybrid RAG, Rankify and strict.
2. Per generator on RAMDocs, FC / Inc of the final policy on the hedging qids above, and the
   decision mix (answered / unresolved / insufficient / needs) per dataset.
3. Save the code as `patches/pipeline_final.patch` (`diff -u` against the original
   `gov_rag_gemini.py`) with a `patches/README.md` that states what A, B and E do and the numbers
   in this brief. The paper's appendix will print the final prompt verbatim, so keep the prompt
   text clean.

I pull with:

```
rsync -av h200:/mnt/data1/gnajpand/conflictbench_h200/results_external/ ~/Documents/Conflicting_documents/results_external/ --include='*_answerandflag_final_*' --exclude='*'
```

## What done looks like

Six judged runs under the `_final` tag with the thinking setting recorded; the hedging-item
check numbers; the comparison table; the patch and README. No change to baselines, strict,
datasets, decision labels, judge prompts or temperatures. Nothing printed from `.env`.

---

# Revision 1 (26 Nov, after the failed pre-check)

Your pre-check report is right on the cause: A's `unresolved_conflict` wording lets a factual
disagreement count as a tie, Gemma calls 3-against-2 a tie, and E is a prompt sentence that
only applies on the commit path and that Gemma follows about a quarter of the time. Two
changes, then one more pre-check with a hard stop.

## A′: unresolved is evaluative-only

In the final prompt, replace A's definition of `unresolved_conflict` and the related text with:

```
- unresolved_conflict: the competing passages disagree on a matter of opinion,
  recommendation, interpretation or research outcome that can legitimately differ, and no
  governance signal separates them. A disagreement about a fact (a number, date, name, place,
  event or attribute) is never an unresolved conflict.
```

and replace the tie rule (step 4) with:

```
4. Factual disagreements are always resolved: apply steps 2 and 3; if the counts are equal and
   step 3 does not separate the values, commit to the value from the more specific, internally
   consistent passage and set confidence to "low". Never resolve a disagreement from your own
   knowledge: the committed value must appear in a supplied passage. If the passages describe
   different entities that share a name, that is not a disagreement: give each entity its own
   value.
```

Keep A's evaluative branch exactly as it is (both positions presented neutrally in the answer,
`short_answers` empty, dissenting passages cited as `conflicting`). Keep B and keep E's prompt
text from the first part of this brief.

## E′: enforce E in the validator

In `_validate_generation_result` (line ~583 on the Mac), after the existing checks pass, add
one more:

1. Collect the rejected values: for every citation with treatment `conflicting`, the
   normalized claim of that chunk (reuse `normalize_claim` / the claim extraction that
   `diversify_conflicts` already runs; for non-numeric claims use the chunk's extracted
   value string). Add every value in the corroboration summary that does not appear in
   `short_answers`. Drop any value that does appear in `short_answers` (it is committed, not
   rejected).
2. If `decision == "answered"` and the `answer` text contains a rejected value (numbers after
   normalization, strings case-insensitive), the result is invalid with the reason:
   `The answer names a rejected value (<value>). The answer must state the committed value(s)
   only; remove <value> from the answer or move it to one final sentence that says it is
   rejected and why. Keep short_answers unchanged.` This goes through the existing single
   repair turn like any other validation failure.
3. If the repair still names the value and the sentence containing it does not contain
   "reject" or "contradict", salvage: keep the decision, `short_answers`, citations and
   explanation, and rebuild `answer` as the committed values, one sentence per entity when the
   explanation names entities, otherwise `"<value>."`. Record `answer_rebuilt: true` in the
   result dict so it reaches the details CSV, and report the count.

The interface shows the dissent from the `conflicting` citations and the corroboration strip,
so nothing the user sees is removed; the answer field just stops hedging. This check applies to
every generator identically and does not look at gold or planted values.

## Pre-check 2

Same protocol, one pass, no tuning. Two item sets, Gemma, hybrid retrieval,
`answer_and_flag_final` with A′ and E′:

1. The 127 RAMDocs items above. Required: FC at least 50 of 127 (was 0, then 13); among the
   103 items Variant 3 answered, `short_answers` unchanged on at least 85; answers naming a
   `wrong_answers` value without a rejecting sentence at most 15; `answer_rebuilt` on at most
   30.
2. 40 CONFLICTS items of type "Conflicting opinions and research outcomes": the first 40 qids
   where the Gemma A-only run was FC. Required: at least 32 stay `unresolved_conflict` and
   FC. This is the guard that A′ did not lose A's gain.

Report both tables as before, with five PC answers and judge reasons if set 1 falls short.

## Stop rule

If pre-check 2 passes both sets, start the six full runs as planned (Gemma RAMDocs first) and
judge them. If it fails either set, stop for good: do not iterate further, write the patch and
README for the record, and tell me. The paper then uses the current table.
