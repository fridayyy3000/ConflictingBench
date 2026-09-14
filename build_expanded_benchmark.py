#!/usr/bin/env python3
"""Build a flat, self-contained 30-question ConflictBench Easy expansion.

The generated folder contains the original 15 Easy questions/documents plus
15 new questions spanning numeric, date, entity, categorical, and boolean
answers. Ground truth remains separate from inference-time documents.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "conflictbench_fictional_full"
TARGET = ROOT / "conflictbench_fictional_all"


NEW_CASES = [
    {
        "question_id": "Q016",
        "question": "What is the maximum number of concurrent privileged sessions allowed for Helix-9 administrative workspaces at Vaelora Cloud?",
        "answer_type": "numeric",
        "gold_answer": "14 sessions",
        "dominant": "10 sessions",
        "alternates": ["12 sessions", "18 sessions"],
        "company": "Vaelora Cloud",
        "policy": "Privileged Workspace Access Standard",
        "topic": "concurrent privileged-session limits",
        "scope": "Helix-9 administrative workspaces",
        "claim": "For Helix-9 administrative workspaces, the maximum permitted number of concurrent privileged sessions is **{answer}**.",
        "effective_date": "September 8, 2026",
        "gold_position": 4,
        "gold_style": 0,
    },
    {
        "question_id": "Q017",
        "question": "What minimum encryption-key length is required for Cinder-X telemetry archives at Bravion Aerospace?",
        "answer_type": "numeric",
        "gold_answer": "3,072 bits",
        "dominant": "2,048 bits",
        "alternates": ["4,096 bits", "1,536 bits"],
        "company": "Bravion Aerospace",
        "policy": "Telemetry Archive Cryptography Standard",
        "topic": "encryption-key length requirements",
        "scope": "Cinder-X telemetry archives",
        "claim": "For Cinder-X telemetry archives, the minimum required encryption-key length is **{answer}**.",
        "effective_date": "August 19, 2026",
        "gold_position": 9,
        "gold_style": 1,
    },
    {
        "question_id": "Q018",
        "question": "What cooling-off interval is required after sterilizing Batch-N4 vessels at Caldris Pharma?",
        "answer_type": "numeric",
        "gold_answer": "38 minutes",
        "dominant": "30 minutes",
        "alternates": ["45 minutes", "24 minutes"],
        "company": "Caldris Pharma",
        "policy": "Sterilization Recovery Procedure",
        "topic": "post-sterilization cooling intervals",
        "scope": "Batch-N4 vessels",
        "claim": "After sterilization, Batch-N4 vessels require a cooling-off interval of **{answer}** before release.",
        "effective_date": "July 27, 2026",
        "gold_position": 2,
        "gold_style": 2,
    },
    {
        "question_id": "Q019",
        "question": "On what date are Aurora-class emissions reports due at Norwyn Transit?",
        "answer_type": "date",
        "gold_answer": "October 12, 2026",
        "dominant": "October 31, 2026",
        "alternates": ["September 30, 2026", "November 15, 2026"],
        "company": "Norwyn Transit",
        "policy": "Fleet Emissions Reporting Calendar",
        "topic": "emissions-report filing dates",
        "scope": "Aurora-class emissions reports",
        "claim": "Aurora-class emissions reports are due on **{answer}**.",
        "effective_date": "July 2, 2026",
        "gold_position": 11,
        "gold_style": 3,
    },
    {
        "question_id": "Q020",
        "question": "On what date must the annual disaster-recovery exercise for Atlas-Blue systems occur at Peloris Bank?",
        "answer_type": "date",
        "gold_answer": "November 18, 2026",
        "dominant": "December 2, 2026",
        "alternates": ["November 4, 2026", "January 13, 2027"],
        "company": "Peloris Bank",
        "policy": "Technology Resilience Exercise Calendar",
        "topic": "disaster-recovery exercise scheduling",
        "scope": "Atlas-Blue systems",
        "claim": "The annual disaster-recovery exercise for Atlas-Blue systems must occur on **{answer}**.",
        "effective_date": "June 15, 2026",
        "gold_position": 6,
        "gold_style": 4,
    },
    {
        "question_id": "Q021",
        "question": "When does the revised Sentinel-R access standard take effect at Quantara Research?",
        "answer_type": "date",
        "gold_answer": "January 7, 2027",
        "dominant": "December 1, 2026",
        "alternates": ["February 1, 2027", "March 15, 2027"],
        "company": "Quantara Research",
        "policy": "Sentinel-R Access Standard",
        "topic": "access-standard effective dates",
        "scope": "the revised Sentinel-R access standard",
        "claim": "The revised Sentinel-R access standard takes effect on **{answer}**.",
        "effective_date": "September 1, 2026",
        "gold_position": 1,
        "gold_style": 0,
    },
    {
        "question_id": "Q022",
        "question": "Which body has final approval authority for Zephyr-2 production releases at Avenor Systems?",
        "answer_type": "entity",
        "gold_answer": "Platform Reliability Council",
        "dominant": "Release Engineering Team",
        "alternates": ["Security Operations Board", "Product Delivery Office"],
        "company": "Avenor Systems",
        "policy": "Production Release Authority Charter",
        "topic": "production-release approval authority",
        "scope": "Zephyr-2 production releases",
        "claim": "Final approval authority for Zephyr-2 production releases belongs to the **{answer}**.",
        "effective_date": "May 23, 2026",
        "gold_position": 8,
        "gold_style": 1,
    },
    {
        "question_id": "Q023",
        "question": "Which office may grant quarantine waivers for Delta-R specimens at Miravel Bio?",
        "answer_type": "entity",
        "gold_answer": "Biosafety Compliance Office",
        "dominant": "Laboratory Operations Office",
        "alternates": ["Clinical Programs Office", "Facilities Safety Team"],
        "company": "Miravel Bio",
        "policy": "Specimen Quarantine Waiver Policy",
        "topic": "specimen-quarantine waiver authority",
        "scope": "Delta-R specimens",
        "claim": "Quarantine waivers for Delta-R specimens may be granted only by the **{answer}**.",
        "effective_date": "April 11, 2026",
        "gold_position": 3,
        "gold_style": 2,
    },
    {
        "question_id": "Q024",
        "question": "Which group owns the final investigation of P0-Crimson incidents at Rethora Networks?",
        "answer_type": "entity",
        "gold_answer": "Critical Incident Review Board",
        "dominant": "Network Operations Center",
        "alternates": ["Customer Reliability Team", "Infrastructure Audit Office"],
        "company": "Rethora Networks",
        "policy": "Critical Incident Ownership Standard",
        "topic": "critical-incident investigation ownership",
        "scope": "P0-Crimson incidents",
        "claim": "Final investigation ownership for P0-Crimson incidents rests with the **{answer}**.",
        "effective_date": "March 29, 2026",
        "gold_position": 12,
        "gold_style": 3,
    },
    {
        "question_id": "Q025",
        "question": "Which encryption profile is required for Nimbus-4 customer backups at Sorellian Data?",
        "answer_type": "categorical",
        "gold_answer": "AES-256-GCM",
        "dominant": "AES-128-CBC",
        "alternates": ["ChaCha20-Poly1305", "AES-256-CBC"],
        "company": "Sorellian Data",
        "policy": "Customer Backup Encryption Standard",
        "topic": "backup-encryption profiles",
        "scope": "Nimbus-4 customer backups",
        "claim": "Nimbus-4 customer backups must use the **{answer}** encryption profile.",
        "effective_date": "February 17, 2026",
        "gold_position": 5,
        "gold_style": 4,
    },
    {
        "question_id": "Q026",
        "question": "Which storage tier must hold Borealis-7 forensic snapshots at Trivex Security?",
        "answer_type": "categorical",
        "gold_answer": "Glacier-Red",
        "dominant": "Archive-Blue",
        "alternates": ["Vault-Gold", "Standard-Cold"],
        "company": "Trivex Security",
        "policy": "Forensic Snapshot Storage Standard",
        "topic": "forensic-snapshot storage tiers",
        "scope": "Borealis-7 forensic snapshots",
        "claim": "Borealis-7 forensic snapshots must be retained in the **{answer}** storage tier.",
        "effective_date": "January 26, 2026",
        "gold_position": 10,
        "gold_style": 0,
    },
    {
        "question_id": "Q027",
        "question": "What risk classification applies to Lattice-K payment events at Umbralis Finance?",
        "answer_type": "categorical",
        "gold_answer": "Restricted-Critical",
        "dominant": "Confidential-High",
        "alternates": ["Internal-Sensitive", "Restricted-Standard"],
        "company": "Umbralis Finance",
        "policy": "Payment Event Classification Standard",
        "topic": "payment-event risk classifications",
        "scope": "Lattice-K payment events",
        "claim": "Lattice-K payment events carry the **{answer}** risk classification.",
        "effective_date": "December 9, 2025",
        "gold_position": 7,
        "gold_style": 1,
    },
    {
        "question_id": "Q028",
        "question": "May Ardent-5 contractors export classified case records to personal devices at Loxen Legal?",
        "answer_type": "boolean",
        "gold_answer": "No",
        "dominant": "Yes",
        "alternates": ["Only with manager approval", "Only during travel"],
        "company": "Loxen Legal",
        "policy": "Classified Case Record Handling Policy",
        "topic": "personal-device export permissions",
        "scope": "Ardent-5 contractors handling classified case records",
        "claim": "May Ardent-5 contractors export classified case records to personal devices? **{answer}**.",
        "effective_date": "November 20, 2025",
        "gold_position": 4,
        "gold_style": 2,
    },
    {
        "question_id": "Q029",
        "question": "Is approval from two independent reviewers mandatory for Nova-Prime releases at Tyveris AI?",
        "answer_type": "boolean",
        "gold_answer": "Yes",
        "dominant": "No",
        "alternates": ["Only for external releases", "Only after an incident"],
        "company": "Tyveris AI",
        "policy": "Nova-Prime Release Review Standard",
        "topic": "independent release-review requirements",
        "scope": "Nova-Prime releases",
        "claim": "Is approval from two independent reviewers mandatory for Nova-Prime releases? **{answer}**.",
        "effective_date": "October 6, 2025",
        "gold_position": 9,
        "gold_style": 3,
    },
    {
        "question_id": "Q030",
        "question": "Can an offline Orion-Lite model bypass red-team review at Cevora Analytics?",
        "answer_type": "boolean",
        "gold_answer": "No",
        "dominant": "Yes",
        "alternates": ["Only in development", "Only with product-owner approval"],
        "company": "Cevora Analytics",
        "policy": "Model Red-Team Review Policy",
        "topic": "red-team review exemptions",
        "scope": "offline Orion-Lite models",
        "claim": "Can an offline Orion-Lite model bypass red-team review? **{answer}**.",
        "effective_date": "September 14, 2025",
        "gold_position": 2,
        "gold_style": 4,
    },
]


GOLD_STYLES = [
    (
        "Governing Policy",
        "Following final approval by {company}'s governance council, this is the active governing policy for {policy}.\n\n"
        "{claim}\n\nIt supersedes previous operational guidance on {topic}.\n\nEffective date: {effective_date}"
    ),
    (
        "Official Policy Decision",
        "This official policy decision from {company} is currently effective for {scope}.\n\n"
        "{claim}\n\nEarlier summaries remain available for historical reference but do not control current decisions.\n\nEffective date: {effective_date}"
    ),
    (
        "Operational Directive",
        "Governing status: ACTIVE\n\n{company} issued this authoritative directive for {scope}.\n\n"
        "{claim}\n\nThis directive replaces conflicting working material.\n\nEffective date: {effective_date}"
    ),
    (
        "Ratified Standard",
        "This document is active and authoritative for {policy} at {company}.\n\n"
        "{claim}\n\nThe ratified standard controls over handbooks, FAQs, and planning notes.\n\nEffective date: {effective_date}"
    ),
    (
        "Policy Bulletin",
        "This policy bulletin from {company} is effective immediately for {scope}.\n\n"
        "{claim}\n\nIt supersedes earlier guidance concerning {topic}.\n\nIssued: {effective_date}"
    ),
]


def render_gold(case: dict) -> str:
    title, body = GOLD_STYLES[case["gold_style"]]
    claim = case["claim"].format(answer=case["gold_answer"])
    context = {**case, "claim": claim}
    return f"# {title}: {case['policy']}\n\n" + body.format(**context) + "\n"


def render_conflict(case: dict, variant: str, answer: str) -> str:
    claim = case["claim"].format(answer=answer)
    common = {
        "company": case["company"],
        "policy": case["policy"],
        "topic": case["topic"],
        "scope": case["scope"],
        "claim": claim,
    }
    templates = {
        "archived": (
            "Archived Operations Handbook",
            "An earlier {company} handbook records the following instruction for {scope}.\n\n{claim}\n\n"
            "Status: ARCHIVED / SUPERSEDED\n\nThis edition is retained for historical audit purposes and is no longer valid."
        ),
        "draft": (
            "Working Draft",
            "A working group proposed the following update to {policy}.\n\n{claim}\n\n"
            "Status: DRAFT PROPOSAL\n\nThe proposal has not completed approval and must not replace the current policy."
        ),
        "secondary": (
            "Implementation Memo",
            "An implementation memo at {company} summarizes {topic}.\n\n{claim}\n\n"
            "Source type: Secondary internal summary. Consult the active governing policy if a conflict exists."
        ),
        "wrong_scope": (
            "Department Reference",
            "A departmental reference lists the following instruction.\n\n{claim}\n\n"
            "Scope note: This document applies to a neighboring category, not the target category in the current governing policy."
        ),
        "superseded": (
            "Legacy Procedure",
            "A legacy procedure at {company} contains this instruction for {scope}.\n\n{claim}\n\n"
            "This procedure was superseded by a later governing publication and should not be used for current decisions."
        ),
        "faq": (
            "Frequently Asked Questions",
            "A frequently asked questions page gives this short answer about {topic}.\n\n{claim}\n\n"
            "Source type: Secondary internal summary. Consult the active governing policy if a conflict exists."
        ),
        "planning": (
            "Planning Brief",
            "A planning brief models future operations using the following assumption.\n\n{claim}\n\n"
            "Status: PROPOSED\n\nThis planning assumption was discussed but was not adopted as the governing requirement."
        ),
        "training": (
            "Training Guide",
            "A training guide circulated at {company} teaches the following rule for {scope}.\n\n{claim}\n\n"
            "Status: ARCHIVED\n\nThe guide predates the current policy and is maintained only for training-history review."
        ),
    }
    title, body = templates[variant]
    return f"# {title}: {case['policy']}\n\n" + body.format(**common) + "\n"


def render_noise(case: dict, index: int) -> str:
    titles = ["Governance Meeting Notes", "Audit Process Overview", "Change-Control Checklist"]
    bodies = [
        "These meeting notes discuss ownership, review cadence, and escalation paths for {topic} at {company}. They do not state the controlling answer for {scope}.",
        "This overview describes how {company} audits compliance with {policy}, including evidence collection and exception logging. It does not provide the current requirement for {scope}.",
        "This checklist covers drafting, legal review, publication, and employee acknowledgement for changes involving {topic}. It contains no approved rule for {scope}.",
    ]
    return f"# {titles[index]}: {case['policy']}\n\n" + bodies[index].format(**case) + "\n"


def new_document_plan(case: dict) -> list[dict]:
    conflicts = [
        ("archived", case["dominant"]),
        ("draft", case["dominant"]),
        ("secondary", case["dominant"]),
        ("wrong_scope", case["dominant"]),
        ("superseded", case["alternates"][0]),
        ("faq", case["alternates"][0]),
        ("planning", case["alternates"][1]),
        ("training", case["alternates"][1]),
    ]
    plan = [{"role": "conflict", "variant": v, "answer": a} for v, a in conflicts]
    plan.extend({"role": "noise", "variant": f"noise_{i + 1}", "answer": ""} for i in range(3))
    plan.insert(case["gold_position"] - 1, {"role": "gold", "variant": "active_authoritative", "answer": case["gold_answer"]})
    assert len(plan) == 12
    return plan


def infer_original_answer_type(answer: str) -> str:
    return "numeric"


def build() -> None:
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {TARGET}")

    TARGET.mkdir()

    question_rows = []
    manifest_rows = []

    with (SOURCE / "questions.csv").open(newline="", encoding="utf-8") as handle:
        originals = list(csv.DictReader(handle))

    with (SOURCE / "ground_truth_manifest.csv").open(newline="", encoding="utf-8") as handle:
        original_manifest = [row for row in csv.DictReader(handle) if row["difficulty"] == "easy"]

    original_manifest_by_qid = defaultdict(list)
    for row in original_manifest:
        original_manifest_by_qid[row["question_id"]].append(row)

    for row in originals:
        qid = row["question_id"]
        question_rows.append({
            "question_id": qid,
            "question": row["question"],
            "answer_type": infer_original_answer_type(row["gold_answer"]),
            "gold_answer": row["gold_answer"],
            "gold_document": row["easy_gold_document"],
            "documents": "12",
            "dominant_conflicting_answer": row["dominant_conflicting_answer"],
            "company": row["company"],
            "policy": row["policy"],
        })

        for item in original_manifest_by_qid[qid]:
            source_doc = SOURCE / "packs" / "easy" / item["document"]
            shutil.copy2(source_doc, TARGET / item["document"])
            manifest_rows.append({
                "question_id": qid,
                "question": row["question"],
                "answer_type": "numeric",
                "gold_answer": row["gold_answer"],
                "document": item["document"],
                "role": item["role"],
                "variant": item["variant"],
                "answer_supported": item["answer_supported"],
            })

    for case in NEW_CASES:
        plan = new_document_plan(case)
        gold_document = f"{case['question_id']}_source_{case['gold_position']:02d}.md"
        question_rows.append({
            "question_id": case["question_id"],
            "question": case["question"],
            "answer_type": case["answer_type"],
            "gold_answer": case["gold_answer"],
            "gold_document": gold_document,
            "documents": "12",
            "dominant_conflicting_answer": case["dominant"],
            "company": case["company"],
            "policy": case["policy"],
        })

        noise_index = 0
        for index, item in enumerate(plan, start=1):
            filename = f"{case['question_id']}_source_{index:02d}.md"
            if item["role"] == "gold":
                text = render_gold(case)
            elif item["role"] == "conflict":
                text = render_conflict(case, item["variant"], item["answer"])
            else:
                text = render_noise(case, noise_index)
                noise_index += 1
            (TARGET / filename).write_text(text, encoding="utf-8")
            manifest_rows.append({
                "question_id": case["question_id"],
                "question": case["question"],
                "answer_type": case["answer_type"],
                "gold_answer": case["gold_answer"],
                "document": filename,
                "role": item["role"],
                "variant": item["variant"],
                "answer_supported": item["answer"],
            })

    q_fields = [
        "question_id", "question", "answer_type", "gold_answer", "gold_document",
        "documents", "dominant_conflicting_answer", "company", "policy",
    ]
    with (TARGET / "questions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=q_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(question_rows)

    with (TARGET / "questions_for_testing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "question", "answer_type"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in question_rows:
            writer.writerow({key: row[key] for key in ["question_id", "question", "answer_type"]})

    m_fields = [
        "question_id", "question", "answer_type", "gold_answer", "document",
        "role", "variant", "answer_supported",
    ]
    with (TARGET / "ground_truth_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=m_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    result_fields = [
        "question_id", "question", "answer_type", "gold_answer", "gold_document",
        "system_answer", "selected_document", "answer_correct", "gold_selected",
        "conflict_detected", "confidence", "claim_extraction_correct",
        "supporting_evidence_correct", "notes",
    ]
    with (TARGET / "results_template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields, lineterminator="\n")
        writer.writeheader()
        for row in question_rows:
            writer.writerow({key: row.get(key, "") for key in result_fields})

    readme = """# ConflictBench Fictional All

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
"""
    (TARGET / "README.md").write_text(readme, encoding="utf-8")

    report = validate(question_rows, manifest_rows)
    (TARGET / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def validate(question_rows: list[dict], manifest_rows: list[dict]) -> dict:
    errors = []
    ids = [row["question_id"] for row in question_rows]
    expected_ids = [f"Q{i:03d}" for i in range(1, 31)]
    if ids != expected_ids:
        errors.append(f"Question IDs are not contiguous Q001-Q030: {ids}")

    docs = sorted(TARGET.glob("Q*_source_*.md"))
    if len(docs) != 360:
        errors.append(f"Expected 360 documents, found {len(docs)}")

    hashes = defaultdict(list)
    for doc in docs:
        hashes[hashlib.sha256(doc.read_bytes()).hexdigest()].append(doc.name)
    duplicates = [names for names in hashes.values() if len(names) > 1]
    new_duplicates = [
        names for names in duplicates
        if any(int(name[1:4]) >= 16 for name in names)
    ]
    if new_duplicates:
        errors.append(f"Duplicate content found in new documents: {new_duplicates}")

    by_qid = defaultdict(list)
    for row in manifest_rows:
        by_qid[row["question_id"]].append(row)

    for question in question_rows:
        qid = question["question_id"]
        rows = by_qid[qid]
        roles = Counter(row["role"] for row in rows)
        if roles != Counter({"conflict": 8, "noise": 3, "gold": 1}):
            errors.append(f"{qid}: wrong role counts {dict(roles)}")
        if len(rows) != 12:
            errors.append(f"{qid}: expected 12 manifest rows, found {len(rows)}")
        gold_rows = [row for row in rows if row["role"] == "gold"]
        if len(gold_rows) == 1:
            gold = gold_rows[0]
            if gold["document"] != question["gold_document"]:
                errors.append(f"{qid}: questions.csv gold document does not match manifest")
            gold_text = (TARGET / gold["document"]).read_text(encoding="utf-8")
            if question["gold_answer"].lower() not in gold_text.lower():
                errors.append(f"{qid}: gold answer missing from gold document")
        for row in rows:
            path = TARGET / row["document"]
            if not path.exists():
                errors.append(f"{qid}: missing document {row['document']}")
                continue
            if row["role"] != "noise" and row["answer_supported"]:
                text = path.read_text(encoding="utf-8").lower()
                if row["answer_supported"].lower() not in text:
                    errors.append(f"{qid}: supported answer missing from {row['document']}")

    if errors:
        raise RuntimeError("Dataset validation failed:\n- " + "\n- ".join(errors))

    return {
        "status": "passed",
        "questions": len(question_rows),
        "documents": len(docs),
        "documents_per_question": 12,
        "roles_per_question": {"gold": 1, "conflict": 8, "noise": 3},
        "answer_types": dict(Counter(row["answer_type"] for row in question_rows)),
        "inherited_original_duplicate_groups": len(duplicates),
        "new_duplicate_groups": len(new_duplicates),
        "checks": [
            "contiguous question IDs",
            "exact document count",
            "no duplicate content among newly generated documents",
            "one gold, eight conflicts, and three noise documents per question",
            "gold document mapping consistency",
            "gold answer present in every gold document",
            "manifest-supported answer present in every claim-bearing document",
        ],
        "notes": [
            "The original Q001-Q015 Easy pack is preserved byte-for-byte.",
            "It contains 15 inherited duplicate groups because each original question reuses identical noise-document text three times.",
        ],
    }


if __name__ == "__main__":
    build()
    print(f"Created {TARGET}")
