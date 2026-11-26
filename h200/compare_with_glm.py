#!/usr/bin/env python3
"""Compare a new judge's FC / Inc with the GLM reference on the rows both judged.

    python compare_with_glm.py --results results_external --slug Qwen-Qwen3-32B

Prints one line per run tag with both judges' FC and Inc and the differences, then,
per dataset, whether the two judges rank the methods by FC in the same order.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

REFERENCE_SLUG = "z-ai-glm-5-3-flash"


def load_summaries(results: str, slug: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(results, f"*_judge3_{slug}_*.json")):
        name = os.path.basename(path)
        m = re.match(rf"(?P<tag>.+)_judge3_{re.escape(slug)}_(?P<stamp>\d{{8}}_\d{{6}})\.json$", name)
        if not m:
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        tag = m.group("tag")
        if tag not in out or m.group("stamp") > out[tag]["_stamp"]:
            data["_stamp"] = m.group("stamp")
            out[tag] = data
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_external")
    ap.add_argument("--slug", required=True, help="new judge slug, e.g. Qwen-Qwen3-32B")
    args = ap.parse_args()

    reference = load_summaries(args.results, REFERENCE_SLUG)
    new = load_summaries(args.results, args.slug)
    shared = sorted(set(reference) & set(new))
    if not shared:
        raise SystemExit(f"no run tag judged by both {REFERENCE_SLUG} and {args.slug} under {args.results}")

    print(f"{'run tag':46} {'GLM FC':>7} {'GLM Inc':>8} {'new FC':>7} {'new Inc':>8} {'dFC':>6} {'dInc':>6}  n(new)")
    for tag in shared:
        a, b = reference[tag], new[tag]
        print(f"{tag:46} {a['FC']:7.1f} {a['Inc']:8.1f} {b['FC']:7.1f} {b['Inc']:8.1f} "
              f"{b['FC'] - a['FC']:+6.1f} {b['Inc'] - a['Inc']:+6.1f}  {b.get('n_labelled', '?')}/{b.get('n', '?')}")

    print()
    for dataset in ("ramdocs", "conflicts"):
        tags = [t for t in shared if t.startswith(dataset + "_")]
        if len(tags) < 2:
            continue
        order_ref = sorted(tags, key=lambda t: -reference[t]["FC"])
        order_new = sorted(tags, key=lambda t: -new[t]["FC"])
        same = order_ref == order_new
        print(f"{dataset}: ranking by FC {'matches' if same else 'DIFFERS'} between judges")
        if not same:
            print("   GLM :", " > ".join(t.replace(dataset + "_nvidia_", "") for t in order_ref))
            print("   new :", " > ".join(t.replace(dataset + "_nvidia_", "") for t in order_new))

    only_new = sorted(set(new) - set(reference))
    if only_new:
        print(f"\njudged by {args.slug} only (no GLM reference): " + ", ".join(only_new))


if __name__ == "__main__":
    main()
