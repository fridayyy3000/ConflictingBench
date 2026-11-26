#!/usr/bin/env python3
"""Evaluation charts for the DocResolve paper, from results_external/.

    python paper/figures/make_charts.py            # writes PDF + PNG next to this file

Reads the newest metrics JSON and details CSV per run tag (main generator only)
and produces:
  fig_tradeoff_ramdocs   RAMDocs set-EM vs misinformation-in-answer, one point per system
  fig_decisions          decision mix per system, RAMDocs and CONFLICTS
  fig_slices_ramdocs     RAMDocs set-EM on items with / without planted misinformation
  fig_conflict_types     CONFLICTS accuracy by annotated conflict type x system (heatmap)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results_external")
GENERATOR = "nvidia/nemotron-3.5-super-vl-120b-a12b"

# ---- systems: tag suffix -> (label, family) ------------------------------
SYSTEMS = {
    "govrag_hybrid_answerandflag": ("DocResolve (hybrid, answer-and-flag)", "flag"),
    "govrag_dense_answerandflag":  ("dense, answer-and-flag", "flag"),
    "govrag_hybrid_strict":        ("hybrid, refuse-on-conflict", "refuse"),
    "govrag":                      ("dense, refuse-on-conflict", "refuse"),
    "hybrid":                      ("hybrid RAG", "other"),
    "baseline":                    ("dense RAG", "other"),
    "closedbook":                  ("closed-book", "other"),
    "llamaindex_k2":               ("LlamaIndex k=2", "other"),
    "llamaindex_k8":               ("LlamaIndex k=8", "other"),
    "rankify":                     ("Rankify", "other"),
    "openwebui":                   ("Open WebUI", "other"),
}
ORDER = list(SYSTEMS)

# ---- palette (validated with the dataviz skill's validator, light mode) ---
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4
GRAY = "#6b7280"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e7eb"
FAMILY_COLOR = {"flag": C1, "refuse": C2, "other": C3}   # first three slots validate all-pairs
FAMILY_LABEL = {"flag": "answer-and-flag", "refuse": "refuse-on-conflict", "other": "controls / off-the-shelf"}
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
COL_W, FULL_W = 3.15, 6.6   # ACL column / text width in inches


def newest(pattern: str) -> dict:
    """{tag_suffix: path} for the newest file per tag matching pattern."""
    out = {}
    for path in glob.glob(os.path.join(RESULTS, pattern)):
        name = os.path.basename(path)
        m = re.match(r"(ramdocs|conflicts)_nvidia_(.+?)_(metrics|details)_(\d{8}_\d{6})\.(json|csv)$", name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if key not in out or m.group(4) > out[key][0]:
            out[key] = (m.group(4), path)
    return {k: v[1] for k, v in out.items()}


def load_metrics() -> dict:
    data = {}
    for (dataset, tag), path in newest("*_metrics_*.json").items():
        m = json.load(open(path))
        if m.get("generation_model") != GENERATOR or tag not in SYSTEMS:
            continue
        data[(dataset, tag)] = m
    return data


def load_details(dataset: str, tag: str) -> pd.DataFrame | None:
    path = newest(f"{dataset}_nvidia_{tag}_details_*.csv").get((dataset, tag))
    if not path:
        return None
    df = pd.read_csv(path)
    if "qid" in df.columns:
        df = df.drop_duplicates("qid", keep="last")
    return df


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "1.0"})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- Fig 5: trade-off scatter ---------------------------------------------
def fig_tradeoff(metrics: dict) -> None:
    pts = []
    for tag in ORDER:
        m = metrics.get(("ramdocs", tag))
        if not m or m.get("set_em") is None or m.get("misinfo_in_answer_rate") is None:
            continue
        pts.append((tag, m["set_em"], m["misinfo_in_answer_rate"]))
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    # label offsets in points and horizontal alignment, tuned against the actual values
    offsets = {"govrag_hybrid_answerandflag": (6, -3, "left"), "govrag_dense_answerandflag": (0, -10, "center"),
               "hybrid": (6, -2, "left"), "baseline": (-6, -4, "right"), "rankify": (6, 2, "left"),
               "govrag": (6, -6, "left"), "govrag_hybrid_strict": (6, 6, "left"),
               "llamaindex_k8": (-6, 0, "right"), "openwebui": (6, 0, "left"),
               "closedbook": (6, 2, "left"), "llamaindex_k2": (0, -10, "center")}
    for tag, x, y in pts:
        label, fam = SYSTEMS[tag]
        main = tag == "govrag_hybrid_answerandflag"
        ax.scatter(x, y, s=46 if main else 30, color=FAMILY_COLOR[fam], edgecolor="white", linewidth=1.2,
                   zorder=3, marker="D" if fam == "refuse" else ("o" if fam == "flag" else "s"))
        dx, dy, ha = offsets.get(tag, (6, 2, "left"))
        ax.annotate(label.replace(" (hybrid, answer-and-flag)", ""), (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.6, color=INK if main else INK2,
                    fontweight="bold" if main else "normal", ha=ha, va="center")
    ax.set_xlim(4, 58); ax.set_ylim(0, 47)
    ax.set_xlabel("set exact match (%)  → better")
    ax.set_ylabel("misinformation in answer (%)  ← better")
    for fam in ("flag", "refuse", "other"):
        ax.scatter([], [], color=FAMILY_COLOR[fam], label=FAMILY_LABEL[fam],
                   marker="D" if fam == "refuse" else ("o" if fam == "flag" else "s"), s=24)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    save(fig, "fig_tradeoff_ramdocs")


# ---- Fig 6: decision mix -------------------------------------------------
DECISIONS = [("answered", "answered", C1), ("unresolved_conflict", "unresolved conflict", C2),
             ("insufficient_evidence", "insufficient evidence", C3), ("needs_information", "needs information", C4)]


def fig_decisions(metrics: dict) -> None:
    rows = [t for t in ORDER if SYSTEMS[t][1] != "other" or t in ("hybrid", "baseline", "closedbook")]
    labels = [SYSTEMS[t][0].replace(" (hybrid, answer-and-flag)", "") for t in rows]
    y = list(range(len(rows)))[::-1]          # same fixed rows in both panels
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    for ax, dataset in zip(axes, ("ramdocs", "conflicts")):
        left = [0.0] * len(rows)
        for key, name, color in DECISIONS:
            vals = []
            for tag in rows:
                m = metrics.get((dataset, tag))
                if not m or not m.get("decisions"):
                    vals.append(0.0)          # run not finished: empty row
                    continue
                vals.append(100.0 * m["decisions"].get(key, 0) / m["total"])
            ax.barh(y, vals, left=left, color=color, height=0.62, edgecolor="white", linewidth=1.0, label=name)
            for yi, l, v in zip(y, left, vals):
                if v >= 7:
                    ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if color in (C1, C2) else INK)
            left = [l + v for l, v in zip(left, vals)]
        for yi, tag in zip(y, rows):
            if not metrics.get((dataset, tag), {}).get("decisions"):
                ax.text(1, yi, "run in progress", va="center", fontsize=6.5, color=INK2, style="italic")
        ax.set_yticks(y); ax.set_yticklabels(labels if ax is axes[0] else [""] * len(rows))
        ax.set_xlim(0, 100); ax.set_xlabel("share of questions (%)")
        ax.set_title("RAMDocs (500)" if dataset == "ramdocs" else "CONFLICTS (458)", loc="left")
        ax.grid(True, axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.08)
    axes[1].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(-0.08, -0.28), handlelength=1.0)
    save(fig, "fig_decisions")


# ---- Fig 7: misinformation slice ----------------------------------------
def fig_slices() -> None:
    rows = ["govrag_hybrid_answerandflag", "govrag_dense_answerandflag", "govrag_hybrid_strict", "govrag",
            "hybrid", "baseline", "rankify", "llamaindex_k8"]
    data = []
    for tag in rows:
        df = load_details("ramdocs", tag)
        if df is None or "set_em_correct" not in df.columns or "wrong_answers" not in df.columns:
            continue
        has_mis = df["wrong_answers"].fillna("").astype(str).str.strip().ne("")
        if not has_mis.any():
            continue
        correct = truthy(df["set_em_correct"])
        data.append((SYSTEMS[tag][0].replace(" (hybrid, answer-and-flag)", ""),
                     100 * correct[has_mis].mean(), 100 * correct[~has_mis].mean(), int(has_mis.sum()), int((~has_mis).sum())))
    if not data:
        print("fig_slices: no rows"); return
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = range(len(data)); w = 0.38
    mis = [d[1] for d in data]; clean = [d[2] for d in data]
    ax.bar([i - w / 2 for i in x], mis, w, color=C2, label=f"with planted misinformation (n={data[0][3]})", edgecolor="white", linewidth=0.8)
    ax.bar([i + w / 2 for i in x], clean, w, color=C1, label=f"without (n={data[0][4]})", edgecolor="white", linewidth=0.8)
    for i, (m, c) in enumerate(zip(mis, clean)):
        ax.text(i - w / 2, m + 1, f"{m:.0f}", ha="center", fontsize=6.5, color=INK2)
        ax.text(i + w / 2, c + 1, f"{c:.0f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in data], rotation=28, ha="right")
    ax.set_ylabel("set exact match (%)"); ax.set_ylim(0, max(clean + mis) + 12)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    save(fig, "fig_slices_ramdocs")


# ---- Fig 8: conflict-type heatmap ----------------------------------------
def fig_conflict_types() -> None:
    rows = ["govrag_hybrid_answerandflag", "govrag_dense_answerandflag", "govrag_hybrid_strict", "govrag",
            "hybrid", "baseline", "closedbook", "llamaindex_k8", "rankify"]
    table = {}
    counts = None
    for tag in rows:
        df = load_details("conflicts", tag)
        if df is None or "conflict_type" not in df.columns or "answer_correct" not in df.columns:
            continue
        if "has_reference_answer" in df.columns:   # containment is defined only where a reference exists
            df = df[truthy(df["has_reference_answer"])]
        ok = truthy(df["answer_correct"])
        grp = ok.groupby(df["conflict_type"].astype(str)).mean() * 100
        table[SYSTEMS[tag][0].replace(" (hybrid, answer-and-flag)", "")] = grp
        if counts is None:
            counts = df["conflict_type"].astype(str).value_counts()
    if not table:
        print("fig_conflict_types: no rows"); return
    mat = pd.DataFrame(table).T
    mat = mat[counts.sort_values(ascending=False).index.intersection(mat.columns)]
    fig, ax = plt.subplots(figsize=(COL_W, 0.28 * len(mat) + 0.9))
    im = ax.imshow(mat.values, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in mat.columns], rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if v > 60 else INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("answer containment (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "fig_conflict_types")


if __name__ == "__main__":
    metrics = load_metrics()
    if not metrics:
        sys.exit(f"no metrics under {RESULTS}")
    fig_tradeoff(metrics)
    fig_decisions(metrics)
    fig_slices()
    fig_conflict_types()
