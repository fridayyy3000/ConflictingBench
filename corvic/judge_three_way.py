#!/usr/bin/env python3
"""Three-way LLM-judge (FC / PC / Inc) over finished evaluate_external.py runs.

Reads one or more ``*_details_*.csv`` files, grades every row's predicted answer
against the gold answer(s) and, for CONFLICTS, the expected behaviour for its
conflict type, and writes per-item labels plus a summary JSON per run.

Labels follow the SentryLine convention:
  FC  fully correct     PC  partially correct     Inc  incorrect

Usage:
  python judge_three_way.py ../results_external/ramdocs_nvidia_govrag_details_X.csv \
      [more csvs] [--judge-model google/gemma-4-31b-it] [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from evaluate_conflictbench_all import load_dotenv_file
from evaluate_external import CONFLICTS_EXPECTED

LABELS = ("FC", "PC", "Inc")

JUDGE_SYSTEM = """You are an impartial grader for a question-answering system that reads several retrieved documents, some of which may disagree. You compare the system's answer with the gold answer(s) and grade it with exactly one label:

FC  (fully correct): every gold answer is given as the answer, no incorrect value is presented as the answer or as an equally valid possibility, and the required behaviour for the situation is followed.
PC  (partially correct): the answer contains the gold answer(s) but is incomplete, hedges between the gold answer and an incorrect value without committing, or follows the required behaviour only partly. Nothing stated is factually wrong.
Inc (incorrect): the answer commits to an incorrect value, contradicts the gold answer, fails to give any gold answer, refuses or abstains although the gold answer was available, or clearly violates the required behaviour.

Grade substance only, not style or length. Minor differences in wording, formatting, units or precision that preserve the meaning do not count against the answer. Return JSON only, on one line: {"label": "FC" | "PC" | "Inc", "reason": "one sentence"}"""

RAMDOCS_RUBRIC = (
    "Situation: the retrieved documents may mix correct information, misinformation and noise. "
    "If the question is ambiguous because several distinct entities share a name, each gold answer "
    "belongs to one entity and all of them must be given for FC. Known misinformation values are "
    "listed; presenting one of them as the answer is Inc, presenting it as an unresolved alternative "
    "alongside the gold answer is PC."
)


def _parse_label(raw: str) -> Optional[Dict]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        try:
            from json_repair import repair_json
            obj = repair_json(m.group(0), return_objects=True)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(obj, dict):
        return None
    label = str(obj.get("label", "")).strip()
    norm = {"fc": "FC", "pc": "PC", "inc": "Inc", "incorrect": "Inc",
            "fully correct": "FC", "partially correct": "PC"}
    label = norm.get(label.lower(), label)
    if label not in LABELS:
        return None
    return {"label": label, "reason": str(obj.get("reason", "")).strip()}


class Judge:
    def __init__(self, model: str, base_url: str, max_retries: int = 10, max_tokens: int = 4096):
        from openai import OpenAI
        key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise SystemExit("No NVIDIA_API_KEY in the environment or corvic/.env")
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def _chat(self, user_prompt: str, reminder: bool = False) -> str:
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user_prompt}]
        if reminder:
            messages.append({"role": "user", "content": "Reply with the JSON object only."})
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0, max_tokens=self.max_tokens)
                content = r.choices[0].message.content or ""
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
                if not content.strip():
                    raise RuntimeError(f"empty completion (finish_reason={r.choices[0].finish_reason})")
                return content
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                if (status is not None and status not in (408, 409, 429, 500, 502, 503, 504)) \
                        or attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def grade(self, prompt: str) -> Dict:
        raw = self._chat(prompt)
        parsed = _parse_label(raw)
        if parsed is None:
            raw = self._chat(prompt, reminder=True)
            parsed = _parse_label(raw)
        if parsed is None:
            return {"label": None, "reason": f"unparseable: {raw[:200]}"}
        return parsed


def build_prompt(row: Dict, dataset: str) -> str:
    gold = [g.strip() for g in (row.get("gold_answers") or "").split("|") if g.strip()]
    wrong = [w.strip() for w in (row.get("wrong_answers") or "").split("|") if w.strip()]
    answer = (row.get("predicted_answer") or "").strip() or "(no answer given)"
    decision = row.get("decision") or ""
    parts = [f"Question: {row['question']}"]
    if dataset == "ramdocs":
        parts.append(RAMDOCS_RUBRIC)
        parts.append("Gold answer(s): " + (" | ".join(gold) if gold else "(none)"))
        if wrong:
            parts.append("Known misinformation value(s): " + " | ".join(sorted(set(wrong))))
    else:
        ctype = row.get("conflict_type") or ""
        _, behaviour = CONFLICTS_EXPECTED.get(ctype, ("", ""))
        parts.append(f"Situation: {ctype}. Required behaviour: {behaviour}")
        if gold:
            parts.append("Gold answer: " + gold[0])
        else:
            parts.append("Gold answer: (no reference answer; grade the required behaviour and factual "
                         "consistency with the situation only)")
    if decision and decision != "free_text":
        parts.append(f"System decision label: {decision}")
    parts.append(f"System answer:\n{answer}")
    parts.append("Grade the system answer.")
    return "\n".join(parts)


def pct(flags: List[bool]) -> float:
    return round(100.0 * sum(flags) / len(flags), 2) if flags else 0.0


def summarise(rows: List[Dict], dataset: str, key_fn, name: str) -> Dict:
    out = {}
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    for k, sub in sorted(groups.items()):
        labelled = [r for r in sub if r["label"] in LABELS]
        out[k] = {"n": len(sub), "n_labelled": len(labelled),
                  **{lab: pct([r["label"] == lab for r in labelled]) for lab in LABELS}}
    return {name: out}


def process_csv(path: Path, judge: Judge, args, out_dir: Path) -> Path:
    stem = path.stem  # e.g. ramdocs_nvidia_govrag_details_20261119_142224
    m = re.match(r"(?P<tag>.+)_details_(?P<stamp>\d{8}_\d{6})$", stem)
    if not m:
        raise SystemExit(f"unexpected details filename: {path.name}")
    tag, stamp = m.group("tag"), m.group("stamp")
    dataset = tag.split("_")[0]
    judge_slug = re.sub(r"[^A-Za-z0-9]+", "-", judge.model).strip("-")

    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    checkpoint = out_dir / f"{tag}_judge3_{judge_slug}_checkpoint.jsonl"
    done: Dict[str, Dict] = {}
    if checkpoint.exists() and not args.no_resume:
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("label") in LABELS:
                    done[rec["qid"]] = rec
    todo = [r for r in rows if r["qid"] not in done]
    print(f"\n{path.name}: {len(rows)} rows, {len(done)} resumed, {len(todo)} to judge "
          f"with {judge.model}", file=sys.stderr)

    ckpt = checkpoint.open("a", encoding="utf-8")
    progress = tqdm(total=len(rows), initial=len(done), desc=f"{tag}/judge3", unit="q",
                    file=sys.stderr, dynamic_ncols=True, mininterval=5.0)

    def work(r: Dict) -> Dict:
        t0 = time.time()
        try:
            res = judge.grade(build_prompt(r, dataset))
        except Exception as exc:  # noqa: BLE001
            res = {"label": None, "reason": f"error: {str(exc)[:200]}"}
        return {"qid": r["qid"], **res, "judge_latency_s": round(time.time() - t0, 2)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, r): r for r in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            done[rec["qid"]] = rec
            ckpt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt.flush()
            progress.update(1)
    progress.close()
    ckpt.close()

    merged = []
    for r in rows:
        j = done.get(r["qid"], {"label": None, "reason": "missing"})
        merged.append({**r, "judge3_label": j.get("label"), "judge3_reason": j.get("reason"),
                       "label": j.get("label")})

    out_csv = out_dir / f"{tag}_judge3_{judge_slug}_{stamp}.csv"
    fields = list(rows[0].keys()) + ["judge3_label", "judge3_reason"]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow(r)

    labelled = [r for r in merged if r["label"] in LABELS]
    summary = {
        "source_details": str(path), "dataset": dataset, "tag": tag, "stamp": stamp,
        "judge_model": judge.model, "n": len(merged), "n_labelled": len(labelled),
        "n_unlabelled": len(merged) - len(labelled),
        **{lab: pct([r["label"] == lab for r in labelled]) for lab in LABELS},
        "mean_judge_latency_s": round(
            sum(done[r["qid"]].get("judge_latency_s", 0) for r in merged if r["qid"] in done) / max(1, len(done)), 2),
    }
    if dataset == "ramdocs":
        summary.update(summarise(merged, dataset, lambda r: f"gold_{r['num_gold_answers']}", "by_num_gold"))
        summary.update(summarise(merged, dataset,
                                 lambda r: "with_misinfo" if (r.get("wrong_answers") or "").strip() else "no_misinfo",
                                 "by_misinfo"))
    else:
        summary.update(summarise(merged, dataset, lambda r: r["conflict_type"], "by_conflict_type"))
        summary.update(summarise(merged, dataset,
                                 lambda r: "with_reference" if (r.get("gold_answers") or "").strip() else "no_reference",
                                 "by_reference"))
    summary.update(summarise(merged, dataset, lambda r: r["decision"], "by_decision"))
    out_json = out_dir / f"{tag}_judge3_{judge_slug}_{stamp}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out_csv.name}\n  -> {out_json.name}\n  FC {summary['FC']}  PC {summary['PC']}  "
          f"Inc {summary['Inc']}  (unlabelled {summary['n_unlabelled']})", file=sys.stderr)
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("details", nargs="+", help="*_details_*.csv files from evaluate_external.py")
    ap.add_argument("--judge-model", default="google/gemma-4-31b-it")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096, help="judge completion budget incl. reasoning")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N rows (smoke test)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out-dir", default=None, help="default: next to the details CSV")
    args = ap.parse_args()

    load_dotenv_file(Path(__file__).resolve().parent / ".env")
    judge = Judge(args.judge_model, args.base_url, max_tokens=args.max_tokens)
    for p in args.details:
        path = Path(p).resolve()
        out_dir = Path(args.out_dir).resolve() if args.out_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        process_csv(path, judge, args, out_dir)


if __name__ == "__main__":
    main()
