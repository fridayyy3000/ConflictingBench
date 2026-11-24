#!/usr/bin/env python3
"""Assemble the SentryLine-style FC / Inc table from *_judge3_*.json summaries.

Usage: python build_judge_table.py [--results ../results_external] [--judge-slug google-gemma-4-31b-it]
Prints Markdown and writes judge3_table_<slug>.md next to the summaries.
"""
import argparse
import glob
import json
import os
import re

METHOD_NAMES = {
    "closedbook": "Direct CoT (few-shot, no retrieval)",
    "baseline": "Vanilla RAG",
    "hybrid": "Hybrid RAG (BM25 + dense, RRF)",
    "llamaindex_k2": "LlamaIndex default (top-k 2)",
    "llamaindex_k8": "LlamaIndex default (top-k 8)",
    "rankify": "Rankify (monoT5 + basic-rag)",
    "openwebui": "Open WebUI (stock document chat)",
    "govrag_hybrid_strict": "GOV-RAG + hybrid retrieval (strict)",
    "govrag_dense_answerandflag": "GOV-RAG answer-and-flag (dense)",
    "govrag_hybrid_answerandflag": "GOV-RAG answer-and-flag + hybrid",
    "govrag": "GOV-RAG (strict, dense; ablation)",
}
MODE_ORDER = ("closedbook", "baseline", "hybrid", "llamaindex_k2", "llamaindex_k8", "rankify", "openwebui", "govrag", "govrag_hybrid_strict", "govrag_dense_answerandflag", "govrag_hybrid_answerandflag")
CONFLICT_SHORT = {
    "No conflict": "No conf.",
    "Complementary information": "Compl.",
    "Conflict due to outdated information": "Outdated",
    "Conflict due to misinformation": "Misinfo",
    "Conflicting opinions and research outcomes": "Opinions",
}


def load(results, slug):
    runs = {}
    for p in glob.glob(os.path.join(results, f"*_judge3_{slug}_*.json")):
        s = json.load(open(p, encoding="utf-8"))
        m = re.match(r"(?P<dataset>\w+?)_(?P<provider>\w+?)_(?P<mode>\w+)$", s["tag"])
        key = (s["dataset"], m.group("mode"), s["judge_model"])
        if key not in runs or s["stamp"] > runs[key]["stamp"]:
            runs[key] = s
    return runs


def cell(d, lab_a="FC", lab_b="Inc"):
    return f"{d[lab_a]:.1f} / {d[lab_b]:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results_external"))
    ap.add_argument("--judge-slug", default="google-gemma-4-31b-it")
    a = ap.parse_args()
    runs = load(a.results, a.judge_slug)
    if not runs:
        raise SystemExit("no judge3 summaries found")
    judge_model = next(iter(runs.values()))["judge_model"]
    modes = [m for m in MODE_ORDER if any(k[1] == m for k in runs)]

    lines = [f"# LLM-judge results (judge: {judge_model}); cells are FC / Inc in %, PC is the remainder\n"]
    # CONFLICTS
    conf = {m: runs.get(("conflicts", m, judge_model)) for m in modes}
    if any(conf.values()):
        types = list(CONFLICT_SHORT)
        lines.append("## CONFLICTS\n")
        lines.append("| Method | All (n) | " + " | ".join(f"{CONFLICT_SHORT[t]} (n)" for t in types) + " | With ref. (n) |")
        lines.append("|---|" + "---|" * (len(types) + 2))
        for m in modes:
            s = conf[m]
            if not s:
                continue
            row = [METHOD_NAMES[m], f"{cell(s)} ({s['n']})"]
            for t in types:
                d = s["by_conflict_type"].get(t)
                row.append(f"{cell(d)} ({d['n']})" if d else "–")
            d = s["by_reference"].get("with_reference")
            row.append(f"{cell(d)} ({d['n']})" if d else "–")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # RAMDocs
    ram = {m: runs.get(("ramdocs", m, judge_model)) for m in modes}
    if any(ram.values()):
        lines.append("## RAMDocs\n")
        lines.append("| Method | All (n) | 1 gold (n) | 2 gold (n) | 3 gold (n) | With misinfo (n) | No misinfo (n) |")
        lines.append("|---|---|---|---|---|---|---|")
        for m in modes:
            s = ram[m]
            if not s:
                continue
            row = [METHOD_NAMES[m], f"{cell(s)} ({s['n']})"]
            for g in ("gold_1", "gold_2", "gold_3"):
                d = s["by_num_gold"].get(g)
                row.append(f"{cell(d)} ({d['n']})" if d else "–")
            for g in ("with_misinfo", "no_misinfo"):
                d = s["by_misinfo"].get(g)
                row.append(f"{cell(d)} ({d['n']})" if d else "–")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # full three-way
    lines.append("## Full FC / PC / Inc\n")
    lines.append("| Dataset | Method | n | FC | PC | Inc | unlabelled |")
    lines.append("|---|---|---|---|---|---|---|")
    for (ds, m, _), s in sorted(runs.items()):
        lines.append(f"| {ds} | {METHOD_NAMES.get(m, m)} | {s['n']} | {s['FC']:.1f} | {s['PC']:.1f} | {s['Inc']:.1f} | {s['n_unlabelled']} |")
    text = "\n".join(lines)
    print(text)
    out = os.path.join(a.results, f"judge3_table_{a.judge_slug}.md")
    open(out, "w", encoding="utf-8").write(text + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
