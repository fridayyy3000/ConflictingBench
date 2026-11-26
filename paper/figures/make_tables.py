#!/usr/bin/env python3
"""Emit the LaTeX bodies of the result tables from results_external/.

    python paper/figures/make_tables.py            # full set
    python paper/figures/make_tables.py --split test   # minus the seeded dev ids

Writes paper/tables/table_main.tex   (judge FC / Inc, three generators)
       paper/tables/table_auto.tex   (automatic metrics, appendix)
Each file is a tabular body only; the \\begin{table*}, caption and label live in
outline.tex.  Newest file per (dataset, generator, tag) wins.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results_external")
OUT = os.path.join(ROOT, "paper", "tables")
JUDGE = "openai-gpt-oss-120b"

GENERATORS = [("nvidia", r"\genA"), ("qwen38-27b", r"\genB"), ("gemma4-31b", r"\genC")]
DATASETS = ["ramdocs", "conflicts"]

# (tag, label, group).  The first tag that exists for the system row is used.
SYSTEM_TAGS = ["govrag_hybrid_answerandflaga"]
ROWS = [
    ("__system__", r"\textbf{\sysname{}} (hybrid retrieval, answer-and-flag)", None),
    ("govrag_hybrid_answerandflag", "answer-and-flag, commit on every conflict (v1)", "Ablations"),
    ("govrag_dense_answerandflag", "dense retrieval, answer-and-flag (v1)", "Ablations"),
    ("govrag_hybrid_strict", "hybrid retrieval, refuse-on-conflict", "Ablations"),
    ("govrag", "dense retrieval, refuse-on-conflict", "Ablations"),
    ("hybrid", "Hybrid RAG", "Controls (same retriever and generator, no policy)"),
    ("baseline", "Vanilla RAG (dense retrieval)", "Controls (same retriever and generator, no policy)"),
    ("closedbook", "Direct CoT (no retrieval)", "Controls (same retriever and generator, no policy)"),
    ("llamaindex_k2", "LlamaIndex, $k$=2", "Off-the-shelf frameworks"),
    ("llamaindex_k8", "LlamaIndex, $k$=8", "Off-the-shelf frameworks"),
    ("rankify", "Rankify", "Off-the-shelf frameworks"),
]


def newest(pattern: str):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_dev_ids(dataset: str) -> set:
    path = os.path.join(RES, f"dev_split_seed2026_{dataset}.txt")
    if not os.path.exists(path):
        raise SystemExit(f"--split test needs {path}")
    return {line.strip() for line in open(path, encoding="utf-8") if line.strip()}


def judge_cell(dataset: str, gen: str, tag: str, dev_ids: set | None):
    """Return (FC, Inc, n) or None."""
    summary = newest(os.path.join(RES, f"{dataset}_{gen}_{tag}_judge3_{JUDGE}_2*.json"))
    if not summary:
        return None
    if dev_ids is None:
        d = json.load(open(summary, encoding="utf-8"))
        return d["FC"], d["Inc"], d["n_labelled"]
    ck = newest(os.path.join(RES, f"{dataset}_{gen}_{tag}_judge3_{JUDGE}_checkpoint.jsonl"))
    det = newest(os.path.join(RES, f"{dataset}_{gen}_{tag}_details_*.csv"))
    if not ck or not det:
        return None
    labels = {}
    for line in open(ck, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        labels[str(rec.get("qid"))] = rec.get("label")
    keep = [labels[str(r["qid"])] for r in csv.DictReader(open(det, encoding="utf-8"))
            if str(r["qid"]) in labels and str(r["qid"]) not in dev_ids]
    if not keep:
        return None
    n = len(keep)
    return 100 * sum(l == "FC" for l in keep) / n, 100 * sum(l == "Inc" for l in keep) / n, n


def auto_cell(dataset: str, gen: str, tag: str, dev_ids: set | None):
    """Automatic metrics: RAMDocs set-EM and misinfo-in-answer; CONFLICTS containment."""
    det = newest(os.path.join(RES, f"{dataset}_{gen}_{tag}_details_*.csv"))
    if not det:
        return None
    rows = list(csv.DictReader(open(det, encoding="utf-8")))
    if dev_ids is not None:
        rows = [r for r in rows if str(r["qid"]) not in dev_ids]
    if not rows:
        return None
    if dataset == "ramdocs":
        em = 100 * sum(r["set_em_correct"] == "True" for r in rows) / len(rows)
        mis = 100 * sum(r["misinfo_in_answer"] == "True" for r in rows) / len(rows)
        return em, mis, len(rows)
    scored = [r for r in rows if r.get("has_reference_answer") == "True"]
    if not scored:
        return None
    acc = 100 * sum(r["answer_correct"] == "True" for r in scored) / len(scored)
    return acc, None, len(scored)


def fmt(x, bold=False):
    if x is None:
        return r"\tbd"
    s = f"{x:.1f}"
    return r"\textbf{" + s + "}" if bold else s


def resolve_system_tag():
    for tag in SYSTEM_TAGS:
        if glob.glob(os.path.join(RES, f"*_{tag}_judge3_{JUDGE}_2*.json")):
            return tag
    return None


def build(cell_fn, two_numbers: bool, dev_split: dict | None):
    system_tag = resolve_system_tag()
    table = {}  # (row_tag) -> {(gen, ds): (a, b, n)}
    for tag, _, _ in ROWS:
        real = system_tag if tag == "__system__" else tag
        table[tag] = {}
        for gen, _ in GENERATORS:
            for ds in DATASETS:
                dev = dev_split[ds] if dev_split else None
                table[tag][(gen, ds)] = cell_fn(ds, gen, real, dev) if real else None
    # best per column: max first number, min second number
    best_a, best_b = {}, {}
    for gen, _ in GENERATORS:
        for ds in DATASETS:
            vals = [(tag, table[tag][(gen, ds)]) for tag, _, _ in ROWS if table[tag][(gen, ds)]]
            if vals:
                best_a[(gen, ds)] = max(v[1][0] for v in vals)
                bs = [v[1][1] for v in vals if v[1][1] is not None]
                best_b[(gen, ds)] = min(bs) if bs else None
    lines = []
    group = None
    for tag, label, grp in ROWS:
        if grp != group and grp is not None:
            lines.append(r"\emph{" + grp + r"} \\")
            group = grp
        cells = []
        for gen, _ in GENERATORS:
            for ds in DATASETS:
                v = table[tag][(gen, ds)]
                if v is None:
                    # mirror the shape of the other rows in this column
                    others = [table[t][(gen, ds)] for t, _, _ in ROWS if table[t][(gen, ds)]]
                    has_second = two_numbers and any(o[1] is not None for o in others)
                    cells.append(r"\tbd" + (r" / \tbd" if has_second else ""))
                    continue
                a, b, _ = v
                s = fmt(a, bold=(a == best_a.get((gen, ds))))
                if two_numbers and b is not None:
                    s += " / " + fmt(b, bold=(b == best_b.get((gen, ds))))
                cells.append(s)
        indent = "" if grp is None else r"\quad "
        lines.append(indent + label + " & " + " & ".join(cells) + r" \\")
        if grp is None:
            lines.append(r"\addlinespace[2pt]")
    return "\n".join(lines) + "\n", system_tag, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["full", "test"], default="full")
    ap.add_argument("--with-auto", action="store_true", help="also emit the automatic-metric table")
    args = ap.parse_args()
    dev_split = {ds: load_dev_ids(ds) for ds in DATASETS} if args.split == "test" else None
    os.makedirs(OUT, exist_ok=True)

    body, system_tag, table = build(judge_cell, two_numbers=True, dev_split=dev_split)
    with open(os.path.join(OUT, "table_main.tex"), "w", encoding="utf-8") as fh:
        fh.write(f"% generated by paper/figures/make_tables.py --split {args.split}; judge {JUDGE}\n")
        fh.write(f"% system row tag: {system_tag or 'NOT YET AVAILABLE'}\n")
        fh.write(body)
    ns = sorted({v[2] for row in table.values() for v in row.values() if v})
    print(f"table_main.tex written (split={args.split}, system={system_tag}, n per cell={ns})")

    # The automatic metrics (set-EM, misinfo-in-answer, containment) are kept
    # out of the paper: one metric vocabulary, the judge's FC / Inc.  Pass
    # --with-auto to emit them anyway for a sanity check.
    if args.with_auto:
        body, _, _ = build(auto_cell, two_numbers=True, dev_split=dev_split)
        with open(os.path.join(OUT, "table_auto.tex"), "w", encoding="utf-8") as fh:
            fh.write(f"% generated by paper/figures/make_tables.py --split {args.split}; automatic metrics\n")
            fh.write(body)
        print("table_auto.tex written")


if __name__ == "__main__":
    main()
