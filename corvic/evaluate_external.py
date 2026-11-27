#!/usr/bin/env python3
"""Evaluate GOV-RAG (and the plain-RAG baseline) on external conflict benchmarks.

Supported datasets (download the raw files into ../data/external/ first):

  ramdocs    RAMDocs test set (Wang et al. 2025, COLM; MIT).
             ../data/external/ramdocs/RAMDocs_test.jsonl
             Per-question documents labelled correct / misinfo / noise, 1-3
             gold answers per question. Official metric: strict set-EM, i.e.
             the response contains ALL gold answers and NO misinformation
             answer. We also report cited-document precision, misinformation
             citation rate, gold-in-context, and a single-gold subset.

  conflicts  CONFLICTS (Cattan et al. 2025, Google; Apache-2.0).
             ../data/external/conflicts/conflicts.jsonl
             Real Google results with dates and a conflict-type label whose
             expected behaviour differs by type. We score (a) whether the
             pipeline's decision matches the type's expected decision,
             (b) answer containment where a correct_answer is provided
             (mostly No-conflict, Outdated and Misinformation), and
             (c) optionally an LLM-judge adherence score (--judge).

Each question is answered against ITS OWN documents (a per-question corpus),
which is how both benchmarks are defined. Ground truth is consulted only
after the prediction exists.

Examples:
  python evaluate_external.py --dataset ramdocs   --provider nvidia --mode govrag   --limit 5
  python evaluate_external.py --dataset conflicts --provider nvidia --mode baseline --sample 100 --seed 1 --judge
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import re
import string
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline.gov_rag_gemini import Document, GovRAGGemini  # noqa: E402
from evaluate_conflictbench_all import ensure_local_copies, load_dotenv_file  # noqa: E402

DATA_DIR = HERE.parent / "data" / "external"
RESULTS_DIR = HERE.parent / "results_external"

# CONFLICTS: expected behaviour per conflict type (paraphrased from the paper's
# taxonomy) and the GOV-RAG decision that corresponds to it.
CONFLICTS_EXPECTED = {
    "No conflict": (
        "answered",
        "The sources agree; the response should give the single supported answer directly.",
    ),
    "Complementary information": (
        "answered",
        "The sources give different but compatible pieces of information (e.g. the answer "
        "differs by place, interpretation or sub-question); the response should answer by "
        "combining or distinguishing them rather than picking one and ignoring the rest.",
    ),
    "Conflicting opinions and research outcomes": (
        "unresolved_conflict",
        "The sources genuinely disagree (opinions, contested research); the response should "
        "present the competing positions neutrally and not commit to one as the answer.",
    ),
    "Conflict due to outdated information": (
        "answered",
        "Some sources are out of date; the response should prefer the most recent information "
        "and give that as the answer, optionally noting that older sources differ.",
    ),
    "Conflict due to misinformation": (
        "answered",
        "Some sources are false; the response should ignore the misinformation and give the "
        "correct answer.",
    ),
}


@dataclass
class ExtDoc:
    doc_id: str
    text: str
    label: str = ""          # ramdocs: correct / misinfo / noise
    answer: str = ""         # ramdocs: the answer this doc supports
    date: str = ""           # conflicts: publication date
    title: str = ""
    url: str = ""


@dataclass
class ExtItem:
    qid: str
    question: str
    documents: List[ExtDoc]
    gold_answers: List[str] = field(default_factory=list)
    wrong_answers: List[str] = field(default_factory=list)
    conflict_type: str = ""
    expected_decision: str = ""
    source: str = ""


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_ramdocs(path: Path) -> List[ExtItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            docs = [
                ExtDoc(doc_id=f"ramdocs_q{i:04d}_d{j:02d}.md", text=d["text"], label=d["type"], answer=d.get("answer", ""))
                for j, d in enumerate(r["documents"])
            ]
            items.append(ExtItem(
                qid=f"q{i:04d}",
                question=r["question"],
                documents=docs,
                gold_answers=list(r.get("gold_answers", [])),
                wrong_answers=list(r.get("wrong_answers", [])),
                expected_decision="answered",
                source="ramdocs",
            ))
    return items


def load_conflicts(path: Path, text_field: str = "short_text") -> List[ExtItem]:
    items = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            docs = []
            for j, s in enumerate(r["search_results"]):
                body = (s.get(text_field) or s.get("short_text") or "").strip()
                if not body:
                    continue  # 675 of ~4,200 results have no extracted text
                header = f"# {s.get('title', '').strip() or 'Web result'}\n\nSource: {s.get('url', '')}\nPublished: {s.get('date', '') or 'unknown'}\n\n"
                docs.append(ExtDoc(
                    doc_id=f"conflicts_q{i:04d}_r{j:02d}.md",
                    text=header + body,
                    date=s.get("date", "") or "",
                    title=s.get("title", ""),
                    url=s.get("url", ""),
                ))
            ctype = r["conflict_type"]
            expected, _ = CONFLICTS_EXPECTED.get(ctype, ("answered", ""))
            answer = (r.get("correct_answer") or "").strip()
            items.append(ExtItem(
                qid=f"q{i:04d}",
                question=r["question"],
                documents=docs,
                gold_answers=[answer] if answer else [],
                conflict_type=ctype,
                expected_decision=expected,
                source=r.get("source", ""),
            ))
    return items


# ----------------------------------------------------------------------------
# Answer matching
# ----------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(text: Optional[str]) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles."""
    text = (text or "").lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains(answer: str, target: str) -> bool:
    a, t = normalize_text(answer), normalize_text(target)
    return bool(t) and t in a


def span_matches(pred: str, gold: str) -> bool:
    """Lenient span equality used for set-EM over short answers.

    True when the normalised strings are equal or one contains the other,
    so "3,559" matches the gold "3,559 people" and "Denali in Alaska" matches
    "Denali". Both sides must be non-empty.
    """
    p, g = normalize_text(pred), normalize_text(gold)
    return bool(p) and bool(g) and (p == g or g in p or p in g)


def gold_found(gold: str, short_answers: List[str], full_answer: str) -> bool:
    if short_answers:
        return any(span_matches(s, gold) for s in short_answers)
    return contains(full_answer, gold)


# Appended to the question for both modes so the task definition matches the
# benchmark. RAMDocs is built on AmbigDocs: several real entities can share
# the name in the question, and the official aggregator prompt asks for all
# correct answers.
TASK_HINTS = {
    "ramdocs": (
        "If the question is ambiguous because several distinct entities share the same name, "
        "give the correct answer for each entity and list each one in short_answers."
    ),
    "conflicts": "",
}


# ----------------------------------------------------------------------------
# Optional LLM judge for CONFLICTS behaviour adherence
# ----------------------------------------------------------------------------

JUDGE_SYSTEM = """You are grading whether a question-answering system's response follows the expected behaviour for its situation. Judge only the behaviour described; do not grade writing quality. Return JSON only: {"adherent": true|false, "reason": "one sentence"}."""


def judge_adherence(chat, item: ExtItem, response_answer: str, decision: str) -> Optional[bool]:
    expected_decision, behaviour = CONFLICTS_EXPECTED.get(item.conflict_type, ("", ""))
    if not behaviour:
        return None
    gold = f"\nReference answer (if any): {item.gold_answers[0]}" if item.gold_answers else ""
    prompt = (
        f"Question: {item.question}\n"
        f"Situation: {item.conflict_type}.\n"
        f"Expected behaviour: {behaviour}{gold}\n\n"
        f"System decision label: {decision}\n"
        f"System response:\n{response_answer}\n\n"
        "Does the response follow the expected behaviour?"
    )
    raw = chat(JUDGE_SYSTEM, prompt)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return bool(json.loads(m.group(0)).get("adherent"))
    except (ValueError, AttributeError):
        return None


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

def build_providers(args):
    if args.provider != "nvidia":
        raise SystemExit("Only --provider nvidia (local embeddings + NVIDIA generation) is wired for external datasets.")
    from pipeline.providers import LocalEmbedder, OpenAICompatGenerator

    embedder = LocalEmbedder(model_name=args.embed_model)
    if args.mode.startswith("llamaindex"):
        from baselines.llamaindex_baseline import LlamaIndexBaseline
        top_k = int(args.mode.rsplit("_k", 1)[1])
        shared = LockedEmbedder(embedder) if args.workers > 1 else embedder
        generator = LlamaIndexBaseline(shared, model=args.generation_model, top_k=top_k)
    elif args.mode == "rankify":
        from baselines.rankify_baseline import RankifyBaseline
        generator = RankifyBaseline(model=args.generation_model, workers=args.workers)
    elif args.mode == "openwebui":
        from baselines.openwebui_baseline import OpenWebUIBaseline
        generator = OpenWebUIBaseline(model=args.generation_model)
    else:
        generator = None if args.no_llm else OpenAICompatGenerator(model=args.generation_model)
    judge = None
    if args.judge and not args.no_llm:
        judge = OpenAICompatGenerator(model=args.judge_model, enable_thinking=False, max_tokens=512)
    return embedder, generator, judge


class LockedEmbedder:
    """Serialise encode() calls so several worker threads can share one model."""

    def __init__(self, inner):
        self.inner = inner
        self.lock = threading.Lock()
        self.model_name = getattr(inner, "model_name", "")
        self.use_vertex = False

    def encode(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        with self.lock:
            return self.inner.encode(texts, task_type=task_type)


GENERATION_FAILED_REASON = "Generation failed; no answer was selected."


def run_item(item: ExtItem, embedder, generator, use_llm: bool, mode: str, task_hint: str = "",
             govrag_retrieval: str = "dense", govrag_policy: str = "strict") -> Dict:
    documents = [Document(filename=d.doc_id, text=d.text) for d in item.documents]
    if mode.startswith("llamaindex") or mode in ("rankify", "openwebui"):
        # External system: free-text answer, so the hint cannot refer to short_answers.
        hint = task_hint.replace(" and list each one in short_answers", "")
        question = f"{item.question}\n\nInstruction: {hint}" if hint else item.question
        return generator.query(question, documents)
    rag = GovRAGGemini(None, embedder=embedder, generator=generator, use_llm=use_llm, documents=documents)
    question = f"{item.question}\n\nInstruction: {task_hint}" if task_hint else item.question
    started = time.perf_counter()
    result = rag.query(question, mode=mode, retrieval=govrag_retrieval, policy=govrag_policy)
    result["_latency_s"] = time.perf_counter() - started
    return result


def score_item(item: ExtItem, result: Dict, dataset: str, judge_fn=None) -> Dict:
    answer = result.get("answer") or ""
    short_answers = [s for s in (result.get("short_answers") or []) if isinstance(s, str)]
    decision = result.get("decision")
    labels = {d.doc_id: d.label for d in item.documents}
    context_files = [c.get("filename") for c in result.get("candidates", [])]
    cited_files = [c.get("filename") for c in result.get("citations", [])]

    row = {
        "qid": item.qid,
        "question": item.question,
        "conflict_type": item.conflict_type,
        "source": item.source,
        "num_docs": len(item.documents),
        "decision": decision,
        "expected_decision": item.expected_decision,
        "decision_match": decision == item.expected_decision,
        "predicted_answer": answer,
        "short_answers": " | ".join(short_answers),
        "gold_answers": " | ".join(item.gold_answers),
        "wrong_answers": " | ".join(item.wrong_answers),
        "selected_document": result.get("source"),
        "num_cited": len(cited_files),
        "validation_error": bool(result.get("validation_error", False)),
        "repair_attempts": int(result.get("repair_attempts", 0) or 0),
        "dropped_citations": int(result.get("dropped_citations", 0) or 0),
        "latency_s": round(result.get("_latency_s", 0.0), 2),
        "reason": result.get("reason"),
    }
    if "conflict_free" in result:  # answer_and_flag_final only; other runs keep their columns
        row["conflict_free"] = bool(result["conflict_free"])
    if "answer_rebuilt" in result:  # answer_and_flag_final only (E', Revision 1)
        row["answer_rebuilt"] = bool(result["answer_rebuilt"])

    if dataset == "ramdocs":
        # Official protocol (MADAM-RAG): the predicted answer set must contain
        # every gold answer and no misinformation answer, after normalisation.
        found = [gold_found(g, short_answers, answer) for g in item.gold_answers]
        all_gold = bool(found) and all(found)
        any_gold = any(found)
        any_wrong = any(gold_found(w, short_answers, answer) for w in item.wrong_answers)
        # Stricter variant: gold must appear verbatim inside the prose answer.
        strict_all_gold = bool(item.gold_answers) and all(contains(answer, g) for g in item.gold_answers)
        correct_docs = {d for d, l in labels.items() if l == "correct"}
        row.update({
            "set_em_correct": all_gold and not any_wrong,
            "strict_set_em_correct": strict_all_gold and not any(contains(answer, w) for w in item.wrong_answers),
            "any_gold_in_answer": any_gold,
            "misinfo_in_answer": any_wrong,
            "num_gold_answers": len(item.gold_answers),
            "gold_doc_in_context": any(f in correct_docs for f in context_files),
            "cited_correct": sum(1 for f in cited_files if labels.get(f) == "correct"),
            "cited_misinfo": sum(1 for f in cited_files if labels.get(f) == "misinfo"),
            "cited_noise": sum(1 for f in cited_files if labels.get(f) == "noise"),
            "selected_label": labels.get(row["selected_document"] or "", ""),
        })
    elif dataset == "conflicts":
        has_ref = bool(item.gold_answers)
        row.update({
            "has_reference_answer": has_ref,
            "answer_correct": gold_found(item.gold_answers[0], short_answers, answer) if has_ref else None,
            "judge_adherent": judge_fn(item, answer, decision) if judge_fn else None,
        })
    return row


def bootstrap_ci(values: List[bool], n_boot: int = 1000, seed: int = 0):
    vals = [1.0 if v else 0.0 for v in values]
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)] * 100, 1), round(means[int(0.975 * n_boot)] * 100, 1)


def pct(values: List[bool]):
    return round(sum(1 for v in values if v) / len(values) * 100, 2) if values else None


def summarize(rows: List[Dict], dataset: str) -> Dict:
    n = len(rows)
    metrics: Dict = {
        "total": n,
        "decisions": dict(Counter(r["decision"] or "none" for r in rows)),
        "decision_match_rate": pct([r["decision_match"] for r in rows]),
        "validation_errors": sum(1 for r in rows if r["validation_error"]),
        "repairs_used": sum(r["repair_attempts"] for r in rows),
        "items_with_dropped_citations": sum(1 for r in rows if r.get("dropped_citations")),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2) if n else None,
    }
    if dataset == "ramdocs":
        em = [r["set_em_correct"] for r in rows]
        single = [r["set_em_correct"] for r in rows if r["num_gold_answers"] == 1]
        cited_total = sum(r["cited_correct"] + r["cited_misinfo"] + r["cited_noise"] for r in rows)
        metrics.update({
            "set_em": pct(em),
            "set_em_ci95": bootstrap_ci(em),
            "strict_set_em": pct([r["strict_set_em_correct"] for r in rows]),
            "set_em_by_num_gold": {
                str(k): pct([r["set_em_correct"] for r in rows if r["num_gold_answers"] == k])
                for k in sorted({r["num_gold_answers"] for r in rows})
            },
            "set_em_single_gold": pct(single),
            "set_em_single_gold_n": len(single),
            "any_gold_in_answer": pct([r["any_gold_in_answer"] for r in rows]),
            "misinfo_in_answer_rate": pct([r["misinfo_in_answer"] for r in rows]),
            "gold_doc_in_context": pct([r["gold_doc_in_context"] for r in rows]),
            "cited_doc_precision": round(sum(r["cited_correct"] for r in rows) / cited_total * 100, 2) if cited_total else None,
            "items_citing_misinfo": sum(1 for r in rows if r["cited_misinfo"] > 0),
            "selected_label": dict(Counter(r["selected_label"] or "none" for r in rows)),
        })
    elif dataset == "conflicts":
        by_type: Dict[str, Dict] = {}
        for ctype in sorted({r["conflict_type"] for r in rows}):
            sub = [r for r in rows if r["conflict_type"] == ctype]
            with_ref = [r for r in sub if r["has_reference_answer"]]
            judged = [r for r in sub if r["judge_adherent"] is not None]
            by_type[ctype] = {
                "n": len(sub),
                "decision_match": pct([r["decision_match"] for r in sub]),
                "decisions": dict(Counter(r["decision"] or "none" for r in sub)),
                "answer_correct": pct([r["answer_correct"] for r in with_ref]),
                "answer_correct_n": len(with_ref),
                "judge_adherent": pct([r["judge_adherent"] for r in judged]),
                "judge_n": len(judged),
            }
        with_ref = [r for r in rows if r["has_reference_answer"]]
        judged = [r for r in rows if r["judge_adherent"] is not None]
        metrics.update({
            "decision_match_ci95": bootstrap_ci([r["decision_match"] for r in rows]),
            "answer_correct": pct([r["answer_correct"] for r in with_ref]),
            "answer_correct_n": len(with_ref),
            "judge_adherent": pct([r["judge_adherent"] for r in judged]),
            "judge_adherent_ci95": bootstrap_ci([r["judge_adherent"] for r in judged]) if judged else None,
            "judge_n": len(judged),
            "by_conflict_type": by_type,
        })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ramdocs", "conflicts"], required=True)
    parser.add_argument("--data-path", default=None, help="Override the raw dataset file")
    parser.add_argument("--provider", choices=["nvidia"], default="nvidia")
    parser.add_argument("--mode", choices=["govrag", "baseline", "hybrid", "closedbook", "llamaindex_k2", "llamaindex_k8", "rankify", "openwebui"], default="govrag",
                        help="govrag | baseline (dense RAG) | hybrid (BM25+dense RRF RAG) | closedbook (few-shot CoT, no retrieval) | "
                             "llamaindex_k2 / llamaindex_k8 (LlamaIndex default pipeline, similarity_top_k 2 or 8) | "
                             "rankify (Rankify toolkit: monoT5 reranker + basic-rag generator) | openwebui (stock Open WebUI over REST)")
    parser.add_argument("--govrag-retrieval", choices=["dense", "hybrid"], default="dense",
                        help="govrag mode only: first-stage retrieval (hybrid = BM25+dense RRF, as the hybrid control)")
    parser.add_argument("--govrag-policy", choices=["strict", "answer_and_flag", "answer_and_flag_v2", "answer_and_flag_a", "answer_and_flag_final", "two_stage"], default="strict",
                        help="govrag mode only: conflict policy (strict = abstain on unresolved; answer_and_flag = commit to the better-corroborated value and flag dissent; "
                             "answer_and_flag_v2 = corroboration only for factual claims, opinions -> neutral unresolved_conflict, factual ties broken by consistency then stated background knowledge; answer_and_flag_a = change A of v2 only (opinions -> neutral unresolved_conflict, corroboration for factual claims, original factual tie rule); answer_and_flag_final = A + B + E (BRIEF_final_policy.md): no refusal on conflict-free evidence, committed answer separated from the dissent; two_stage = plain draft, then a governance audit that keeps the draft unless the evidence gives a reason to change it)")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--judge", action="store_true", help="CONFLICTS: LLM-judge behaviour adherence")
    parser.add_argument("--judge-model", default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    parser.add_argument("--conflicts-text-field", default="short_text", choices=["short_text", "response_str"])
    parser.add_argument("--task-hint", default=None, help="Override the per-dataset instruction appended to each question ('' to disable)")
    parser.add_argument("--limit", type=int, default=None, help="First N items")
    parser.add_argument("--sample", type=int, default=None, help="Random N items (with --seed)")
    parser.add_argument("--qids-file", default=None, help="Run only the qids listed in this file (whitespace-separated)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel questions (API-bound)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore the per-item checkpoint for this dataset/mode/model and start over")
    args = parser.parse_args()

    load_dotenv_file(HERE / ".env")

    default_path = {
        "ramdocs": DATA_DIR / "ramdocs" / "RAMDocs_test.jsonl",
        "conflicts": DATA_DIR / "conflicts" / "conflicts.jsonl",
    }[args.dataset]
    data_path = Path(args.data_path) if args.data_path else default_path
    if not data_path.exists():
        raise SystemExit(f"Dataset file not found: {data_path}")

    ensure_local_copies([data_path])  # iCloud placeholders block reads indefinitely
    items = load_ramdocs(data_path) if args.dataset == "ramdocs" else load_conflicts(data_path, args.conflicts_text_field)
    if args.sample:
        items = random.Random(args.seed).sample(items, min(args.sample, len(items)))
    if args.qids_file:
        wanted = set(Path(args.qids_file).read_text(encoding="utf-8").split())
        items = [it for it in items if it.qid in wanted]
        missing = wanted - {it.qid for it in items}
        if missing:
            raise SystemExit(f"--qids-file: {len(missing)} qids not in the dataset: {sorted(missing)[:5]}")
    if args.limit:
        items = items[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.dataset}_{args.provider}_{args.mode}"
    if args.mode == "govrag" and (args.govrag_retrieval != "dense" or args.govrag_policy != "strict"):
        # Variant runs get their own tag so the strict/dense GOV-RAG files stay the ablation row.
        # answer_and_flag_final keeps its "_final" suffix (tag *_govrag_hybrid_answerandflag_final,
        # as the brief names it); every other policy drops its underscores as before.
        pol_tag = "answerandflag_final" if args.govrag_policy == "answer_and_flag_final" else args.govrag_policy.replace('_', '')
        tag += f"_{args.govrag_retrieval}_{pol_tag}"
    detail_csv = output_dir / f"{tag}_details_{stamp}.csv"
    metrics_json = output_dir / f"{tag}_metrics_{stamp}.json"
    log_path = output_dir / f"{tag}_run_{stamp}.log"
    # Per-item checkpoint: one JSON row per finished question. A restart with
    # the same dataset/mode/model skips the questions already in it.
    model_slug = re.sub(r"[^A-Za-z0-9]+", "-", args.generation_model or "default").strip("-")
    checkpoint = output_dir / f"{tag}_{model_slug}_checkpoint.jsonl"
    resumed_rows: List[Dict] = []
    if checkpoint.exists() and not args.no_resume:
        wanted = {item.qid for item in items}
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("qid") in wanted:
                    resumed_rows.append(row)
        done_qids = {row["qid"] for row in resumed_rows}
        items = [item for item in items if item.qid not in done_qids]

    err = sys.stderr
    print(f"Dataset: {args.dataset} ({len(items) + len(resumed_rows)} items from {data_path.name}; "
          f"{len(resumed_rows)} resumed from checkpoint, {len(items)} to run)", file=err, flush=True)
    print(f"Provider/mode: {args.provider}/{args.mode}  workers={args.workers}", file=err, flush=True)
    print(f"Details CSV: {detail_csv}\nMetrics JSON: {metrics_json}\nRun log: {log_path}", file=err, flush=True)

    rows: List[Dict] = list(resumed_rows)
    with log_path.open("w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
        embedder, generator, judge = build_providers(args)
        shared_embedder = LockedEmbedder(embedder) if args.workers > 1 else embedder
        judge_fn = (lambda item, ans, dec: judge_adherence(judge._chat, item, ans, dec)) if judge else None

        task_hint = TASK_HINTS.get(args.dataset, "") if args.task_hint is None else args.task_hint

        ckpt_lock = threading.Lock()
        ckpt_file = checkpoint.open("a", encoding="utf-8")
        transport_failures = [0]

        def work(index_item):
            index, item = index_item
            result = run_item(item, shared_embedder, generator, not args.no_llm, args.mode, task_hint,
                              govrag_retrieval=args.govrag_retrieval, govrag_policy=args.govrag_policy)
            print(f"\n\n===== {item.qid} =====\n{json.dumps(result, default=str, indent=1)}\n", flush=True)
            row = score_item(item, result, args.dataset, judge_fn)
            # Checkpoint here, in the worker, so finished items survive even if
            # the progress-reporting loop in the main thread stalls or dies.
            # A generation that failed on transport (rate limit, timeout) after
            # all retries is kept in this run's CSV but not checkpointed, so a
            # resumed run retries it instead of keeping a spurious verdict.
            with ckpt_lock:
                if row.get("reason") == GENERATION_FAILED_REASON:
                    transport_failures[0] += 1
                else:
                    ckpt_file.write(json.dumps(row, default=str) + "\n")
                    ckpt_file.flush()
            return index, row

        progress = tqdm(total=len(items) + len(resumed_rows), initial=len(resumed_rows),
                        desc=f"{args.dataset}/{args.mode}", unit="q",
                        file=err, dynamic_ncols=True, mininterval=5.0)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(work, (i, item)) for i, item in enumerate(items)]
            done = len(resumed_rows)
            for fut in as_completed(futures):
                try:
                    index, row = fut.result()
                except Exception as exc:  # noqa: BLE001 - one item must never kill the run
                    # e.g. a 429 that survived every retry inside the provider. The
                    # item is left out of the checkpoint, so a resumed run retries it.
                    with ckpt_lock:
                        transport_failures[0] += 1
                    try:
                        progress.write(f"ITEM FAILED ({type(exc).__name__}): {str(exc)[:200]}", file=err)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                rows.append(row)
                done += 1
                try:
                    progress.update(1)
                except Exception:  # noqa: BLE001 - reporting must never stop the run
                    pass
                head = (
                    f"set_em={'OK' if row.get('set_em_correct') else 'MISS'}" if args.dataset == "ramdocs"
                    else f"type={row['conflict_type'][:22]!r} match={'OK' if row['decision_match'] else 'MISS'}"
                    + (f" ans={'OK' if row['answer_correct'] else 'MISS'}" if row.get("has_reference_answer") else "")
                    + (f" judge={'OK' if row['judge_adherent'] else 'MISS'}" if row.get("judge_adherent") is not None else "")
                )
                try:
                    progress.write(f"{done:03d}/{len(items) + len(resumed_rows)} {row['qid']}: decision={row['decision']} {head} {row['latency_s']:.1f}s", file=err)
                except Exception:  # noqa: BLE001
                    pass
        progress.close()
        ckpt_file.close()
        if transport_failures[0]:
            print(f"WARNING: {transport_failures[0]} items failed generation after retries; they are in this CSV as "
                  f"insufficient_evidence but NOT in the checkpoint. Re-run the same command to retry only those.",
                  file=err, flush=True)

    rows.sort(key=lambda r: r["qid"])
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = summarize(rows, args.dataset)
    metrics.update({
        "dataset": args.dataset, "mode": args.mode, "provider": args.provider,
        "generation_model": getattr(generator, "model", None),
        "judge_model": args.judge_model if judge else None,
        "embed_model": args.embed_model, "sample": args.sample, "seed": args.seed, "limit": args.limit, "qids_file": args.qids_file,
        "task_hint": TASK_HINTS.get(args.dataset, "") if args.task_hint is None else args.task_hint,
        "govrag_retrieval": args.govrag_retrieval if args.mode == "govrag" else None,
        "govrag_policy": args.govrag_policy if args.mode == "govrag" else None,
        "details_csv": str(detail_csv), "run_log": str(log_path),
    })
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSUMMARY", file=err)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("details_csv", "run_log")}, indent=2), file=err)
    print(f"Details CSV: {detail_csv}\nMetrics JSON: {metrics_json}", file=err)


if __name__ == "__main__":
    main()
