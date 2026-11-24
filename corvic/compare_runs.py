#!/usr/bin/env python3
"""Compare two or more ConflictBench evaluation runs side by side.

Usage:
    python compare_runs.py ../results_gemini/conflictbench_all_nvidia_govrag_details_<stamp>.csv \
                           ../results_gemini/conflictbench_all_nvidia_baseline_details_<stamp>.csv

With no arguments, the newest details CSV for each provider/mode pair found in
../results_gemini is used. Prints overall metrics, a breakdown by answer type
(from ground_truth_manifest.csv) and the per-question win/loss table, and
writes the same comparison as Markdown next to the inputs.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results_gemini"
DATASET_DIR = HERE.parent / "conflictbench_fictional_all"

NAME_RE = re.compile(r"conflictbench_all_(?P<tag>.+?)_details_(?P<stamp>\d{8}_\d{6})\.csv$")


def latest_per_tag(results_dir: Path) -> list[Path]:
    newest: dict[str, Path] = {}
    for path in results_dir.glob("conflictbench_all_*_details_*.csv"):
        match = NAME_RE.search(path.name)
        if not match:
            continue
        tag = match.group("tag")
        if tag not in newest or match.group("stamp") > NAME_RE.search(newest[tag].name).group("stamp"):
            newest[tag] = path
    return [newest[tag] for tag in sorted(newest)]


def tag_of(path: Path) -> str:
    match = NAME_RE.search(path.name)
    return match.group("tag") if match else path.stem


def load_details(path: Path) -> OrderedDict[str, dict]:
    rows: OrderedDict[str, dict] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["question_id"]] = row
    return rows


def load_answer_types() -> dict[str, str]:
    manifest = DATASET_DIR / "ground_truth_manifest.csv"
    types: dict[str, str] = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                types.setdefault(row["question_id"], row["answer_type"])
    return types


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def pct(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den * 100:.0f}%)" if den else "n/a"


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or latest_per_tag(RESULTS_DIR)
    if len(paths) < 1:
        raise SystemExit("No details CSVs found.")

    runs = OrderedDict((tag_of(p), load_details(p)) for p in paths)
    answer_types = load_answer_types()
    qids = sorted({q for rows in runs.values() for q in rows})

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("# ConflictBench comparison")
    out()
    for tag, path in zip(runs, paths):
        out(f"- `{tag}`: {path.name}")
    out()

    # Overall metrics
    out("## Overall")
    out()
    header = ["metric"] + list(runs)
    table: list[list[str]] = []
    for label, key in [
        ("answer correct", "answer_correct"),
        ("gold source selected", "gold_selected"),
        ("gold in final context", "gold_in_context"),
    ]:
        row = [label]
        for rows in runs.values():
            row.append(pct(sum(truthy(r.get(key)) for r in rows.values()), len(rows)))
        table.append(row)
    row = ["mean latency (s)"]
    for rows in runs.values():
        lat = [float(r["latency_s"]) for r in rows.values() if r.get("latency_s")]
        row.append(f"{sum(lat) / len(lat):.1f}" if lat else "n/a")
    table.append(row)
    emit_table(out, header, table)

    out("## Decisions")
    out()
    decisions = sorted({r.get("decision") or "none" for rows in runs.values() for r in rows.values()})
    table = []
    for decision in decisions:
        table.append([decision] + [
            str(sum((r.get("decision") or "none") == decision for r in rows.values())) for rows in runs.values()
        ])
    emit_table(out, ["decision"] + list(runs), table)

    out("## Selected document role")
    out()
    roles_by_run: dict[str, Counter] = {}
    role_lookup = load_roles()
    for tag, rows in runs.items():
        counter: Counter = Counter()
        for qid, r in rows.items():
            sel = r.get("selected_document") or ""
            counter[role_lookup.get((qid, sel), "none" if not sel else "unknown")] += 1
        roles_by_run[tag] = counter
    roles = sorted({role for c in roles_by_run.values() for role in c})
    emit_table(out, ["role"] + list(runs), [[role] + [str(roles_by_run[t][role]) for t in runs] for role in roles])

    if answer_types:
        out("## Answer accuracy by answer type")
        out()
        by_type: dict[str, list[str]] = defaultdict(list)
        for qid in qids:
            by_type[answer_types.get(qid, "unknown")].append(qid)
        table = []
        for atype, ids in sorted(by_type.items()):
            row = [f"{atype} (n={len(ids)})"]
            for rows in runs.values():
                row.append(pct(sum(truthy(rows.get(q, {}).get("answer_correct")) for q in ids), len(ids)))
            table.append(row)
        emit_table(out, ["answer type"] + list(runs), table)

    out("## Per question")
    out()
    header = ["qid", "type"] + [f"{t} answer" for t in runs] + [f"{t} source" for t in runs]
    table = []
    for qid in qids:
        row = [qid, answer_types.get(qid, "")]
        for rows in runs.values():
            r = rows.get(qid, {})
            row.append(mark(r, "answer_correct"))
        for rows in runs.values():
            r = rows.get(qid, {})
            row.append(mark(r, "gold_selected") + (f" [{r.get('decision')}]" if r.get("decision") not in (None, "", "answered") else ""))
        table.append(row)
    emit_table(out, header, table)

    report = paths[0].parent / ("comparison_" + "_vs_".join(runs) + ".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMarkdown written to {report}")


def mark(row: dict, key: str) -> str:
    if not row:
        return "-"
    return "OK" if truthy(row.get(key)) else "miss"


def load_roles() -> dict[tuple[str, str], str]:
    manifest = DATASET_DIR / "ground_truth_manifest.csv"
    roles: dict[tuple[str, str], str] = {}
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                roles[(row["question_id"], row["document"])] = row["role"]
    return roles


def emit_table(out, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    out(fmt.format(*header))
    out("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out(fmt.format(*[str(x) for x in row]))
    out()


if __name__ == "__main__":
    main()
