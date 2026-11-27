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
    "govrag_hybrid_answerandflagv2": "GOV-RAG answer-and-flag v2 (A+C) + hybrid",
    "govrag_hybrid_answerandflaga": "GOV-RAG answer-and-flag A-only + hybrid",
    "govrag_hybrid_answerandflag_final": "GOV-RAG answer-and-flag final (A'+B+E+E') + hybrid",
    "govrag": "GOV-RAG (strict, dense; ablation)",
}
MODE_ORDER = ("closedbook", "baseline", "hybrid", "llamaindex_k2", "llamaindex_k8", "rankify", "openwebui", "govrag", "govrag_hybrid_strict", "govrag_dense_answerandflag", "govrag_hybrid_answerandflag", "govrag_hybrid_answerandflagv2", "govrag_hybrid_answerandflaga", "govrag_hybrid_answerandflag_final")
CONFLICT_SHORT = {
    "No conflict": "No conf.",
    "Complementary information": "Compl.",
    "Conflict due to outdated information": "Outdated",
    "Conflict due to misinformation": "Misinfo",
    "Conflicting opinions and research outcomes": "Opinions",
}


GENERATOR_NAMES = {
    "nvidia": "Nemotron 3.5 Super (NVIDIA API)",
    "qwen38-27b": "Qwen3.8-27B (local vLLM, thinking off)",
    "gemma4-31b": "Gemma-4-31B-it (local vLLM, thinking off)",
}
GENERATOR_ORDER = ("nvidia", "qwen38-27b", "gemma4-31b")
TAG_RE = re.compile(r"(?P<dataset>[a-z]+)_(?P<provider>[a-z0-9-]+)_(?P<mode>\w+)$")


def load(results, slug):
    runs = {}
    for p in glob.glob(os.path.join(results, f"*_judge3_{slug}_*.json")):
        s = json.load(open(p, encoding="utf-8"))
        m = TAG_RE.match(s["tag"])
        if not m:
            print(f"skipping unrecognised tag {s['tag']}")
            continue
        key = (s["dataset"], m.group("provider"), m.group("mode"))
        if key not in runs or s["stamp"] > runs[key]["stamp"]:
            runs[key] = s
    return runs


def cell(d, lab_a="FC", lab_b="Inc"):
    return f"{d[lab_a]:.1f} / {d[lab_b]:.1f}"


def gen_sections(lines, runs, gen, modes):
    conf = {m: runs.get(("conflicts", gen, m)) for m in modes}
    if any(conf.values()):
        types = list(CONFLICT_SHORT)
        lines.append("### CONFLICTS\n")
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
    ram = {m: runs.get(("ramdocs", gen, m)) for m in modes}
    if any(ram.values()):
        lines.append("### RAMDocs\n")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results_external"))
    ap.add_argument("--judge-slug", default="openai-gpt-oss-120b")
    ap.add_argument("--paper", action="store_true",
                    help="paper table: one GOV-RAG row (final policy A'+B+E+E' + hybrid, named GOV-RAG); the ablation table "
                         "holds the final row, strict + hybrid, Variant 3, v2 (A+C) and A-only. Writes judge3_paper_table_<slug>.md")
    a = ap.parse_args()
    runs = load(a.results, a.judge_slug)
    if not runs:
        raise SystemExit("no judge3 summaries found")
    judges = sorted({s["judge_model"] for s in runs.values()})
    if len(judges) != 1:
        raise SystemExit(f"more than one judge model under slug {a.judge_slug}: {judges}")
    judge_model = judges[0]
    modes = [m for m in MODE_ORDER if any(k[2] == m for k in runs)]
    if a.paper:
        return paper(a, runs, judge_model, modes)
    gens = [g for g in GENERATOR_ORDER if any(k[1] == g for k in runs)]
    gens += sorted({k[1] for k in runs} - set(gens))

    lines = [f"# LLM-judge results (judge: {judge_model} for every row); cells are FC / Inc in %, PC is the remainder\n"]
    if len(gens) > 1:
        lines.append("## Across generators (All items, FC / Inc)\n")
        for ds, title in (("conflicts", "CONFLICTS"), ("ramdocs", "RAMDocs")):
            lines.append(f"### {title}\n")
            lines.append("| Method | " + " | ".join(GENERATOR_NAMES.get(g, g) for g in gens) + " |")
            lines.append("|---|" + "---|" * len(gens))
            for m in modes:
                cells = [cell(runs[(ds, g, m)]) if (ds, g, m) in runs else "–" for g in gens]
                if any(c != "–" for c in cells):
                    lines.append(f"| {METHOD_NAMES[m]} | " + " | ".join(cells) + " |")
            lines.append("")
    for g in gens:
        lines.append(f"## Generator: {GENERATOR_NAMES.get(g, g)}\n")
        gen_sections(lines, runs, g, modes)
    lines.append("## Full FC / PC / Inc\n")
    lines.append("| Dataset | Generator | Method | n | FC | PC | Inc | unlabelled |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (ds, g, m), s in sorted(runs.items()):
        lines.append(f"| {ds} | {GENERATOR_NAMES.get(g, g)} | {METHOD_NAMES.get(m, m)} | {s['n']} | {s['FC']:.1f} | {s['PC']:.1f} | {s['Inc']:.1f} | {s['n_unlabelled']} |")
    text = "\n".join(lines)
    print(text)
    out = os.path.join(a.results, f"judge3_table_{a.judge_slug}.md")
    open(out, "w", encoding="utf-8").write(text + "\n")
    print(f"\nwritten {out}")


# 2026-11-27 (user decision): the final policy A'+B+E+E' (govrag_hybrid_answerandflag_final) is the
# paper's GOV-RAG system row. Variant 3 is the ablation "answer-and-flag without commitment"; A-only,
# v2 (A+C) and strict + hybrid stay ablation rows. Earlier: A-only (26 Nov), Variant 3 before that.
PAPER_MAIN = "govrag_hybrid_answerandflag_final"
PAPER_ABLATION = ("govrag_hybrid_answerandflag_final", "govrag_hybrid_strict", "govrag_hybrid_answerandflag",
                  "govrag_hybrid_answerandflagv2", "govrag_hybrid_answerandflaga")
ABLATION_NAMES = {"govrag_hybrid_answerandflag_final": "final A'+B+E+E' (paper GOV-RAG row)",
                  "govrag_hybrid_strict": "strict + hybrid",
                  "govrag_hybrid_answerandflag": "Variant 3: answer-and-flag without commitment",
                  "govrag_hybrid_answerandflagv2": "v2 = A + C", "govrag_hybrid_answerandflaga": "A-only"}


def paper(a, runs, judge_model, modes):
    gens = [g for g in GENERATOR_ORDER if any(k[1] == g for k in runs)]
    main_modes = [m for m in modes if not m.startswith("govrag")] + [PAPER_MAIN]
    name = dict(METHOD_NAMES, **{PAPER_MAIN: "**GOV-RAG** (hybrid retrieval, answer-and-flag final)"})
    lines = [f"# Paper table (judge: {judge_model} for every row). Cells: FC / Inc in %; PC is the remainder.\n",
             "Higher FC and lower Inc are better. Avg = unweighted mean over the 6 generator x dataset cells.\n"]

    def table(ms, title, name=name):
        lines.append(f"## {title}\n")
        hdr = [f"{ds} {GENERATOR_NAMES.get(g, g).split(' (')[0]}" for ds in ("CONFLICTS", "RAMDocs") for g in gens]
        lines.append("| Method | " + " | ".join(hdr) + " | Avg FC | Avg Inc |")
        lines.append("|---|" + "---|" * (len(hdr) + 2))
        for m in ms:
            ss = [runs.get((ds, g, m)) for ds in ("conflicts", "ramdocs") for g in gens]
            if not any(ss):
                continue
            cells = [cell(s) if s else "–" for s in ss]
            full = all(ss)
            avg_fc = f"{sum(s['FC'] for s in ss) / len(ss):.1f}" if full else "–"
            avg_inc = f"{sum(s['Inc'] for s in ss) / len(ss):.1f}" if full else "–"
            lines.append(f"| {name.get(m, m)} | " + " | ".join(cells) + f" | {avg_fc} | {avg_inc} |")
        lines.append("")

    table(main_modes, "Main table")
    table(list(PAPER_ABLATION), "Ablation: GOV-RAG policies (hybrid retrieval)", name=dict(name, **ABLATION_NAMES))
    text = "\n".join(lines)
    print(text)
    out = os.path.join(a.results, f"judge3_paper_table_{a.judge_slug}.md")
    open(out, "w", encoding="utf-8").write(text + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
