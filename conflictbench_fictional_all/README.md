# ConflictBench Fictional All

A single flat Easy benchmark containing the original 15 ConflictBench questions and 15 new validated questions.

## Contents

- 30 questions
- 360 Markdown documents (12 per question)
- 1 governing source, 8 conflicting sources, and 3 same-topic noise sources per question
- `questions.csv`: questions and answer key
- `questions_for_testing.csv`: blind question list without answers or source labels
- `ground_truth_manifest.csv`: document roles and supported claims
- `results_template.csv`: manual evaluation sheet, including evidence-quality checks
- `validation_report.json`: structural validation results

## Uploading to DocuInsight

Upload only the 360 `Q*_source_*.md` files. Do not upload the CSV, JSON, or README files because they contain evaluation labels.

The folder is deliberately flat so all documents can be selected in one upload. The total is below DocuInsight's 500-file limit.

## Coverage

- Q001-Q015: original numeric policy-value conflicts
- Q016-Q018: new numeric rules
- Q019-Q021: dates
- Q022-Q024: responsible bodies/offices
- Q025-Q027: categorical requirements
- Q028-Q030: boolean rules

The new cases vary authority language and governing-document position. All claim-bearing documents within a case address the same relation as the question.
