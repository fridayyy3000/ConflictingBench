#!/usr/bin/env python3
"""Evaluate the flat conflictbench_fictional_all folder with GOV-RAG.

Providers
---------
--provider vertex   Gemini embeddings + Gemini 2.5 Pro via Vertex AI (original).
--provider nvidia   Local sentence-transformers embeddings (bge-small, cached)
                    + a Nemotron model on NVIDIA's OpenAI-compatible API.
                    Needs NVIDIA_API_KEY in the environment or in corvic/.env.

Modes
-----
--mode govrag       Full pipeline: authority/scope reranking, claim
                    diversification, governance-aware prompt.
--mode baseline     Plain-RAG control: same retrieval, top-K by semantic score
                    only, passages-only prompt.

Ground truth (gold answer / gold document) is read only after each prediction
is produced; the RAG instance never sees questions.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline.gov_rag_gemini import GovRAGGemini  # noqa: E402


def load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_answer(value: str | None) -> str:
    if not value:
        return ""
    text = value.lower().strip()
    text = text.replace("degrees celsius", "c")
    text = text.replace("°c", "c")
    text = re.sub(r"[,\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def answer_matches(predicted: str | None, gold: str) -> bool:
    pred = normalize_answer(predicted)
    target = normalize_answer(gold)
    if not pred or not target:
        return False
    return pred == target or target in pred


def ensure_local_copies(paths: list[Path], timeout_s: float = 180.0) -> None:
    """Make sure iCloud-synced files are actually on disk before reading them.

    On macOS with "Optimize Mac Storage", files under ~/Documents can be
    evicted to placeholders (size > 0 but zero allocated blocks). Opening one
    blocks until iCloud downloads it, which can hang indefinitely. Request the
    download explicitly and wait, instead of hanging inside the pipeline.
    """
    def evicted() -> list[Path]:
        out = []
        for path in paths:
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_size > 0 and getattr(st, "st_blocks", 1) == 0:
                out.append(path)
        return out

    pending = evicted()
    if not pending:
        return
    print(f"{len(pending)} dataset files are iCloud placeholders; requesting download...", flush=True)
    if shutil.which("brctl"):
        for path in pending:
            subprocess.run(["brctl", "download", str(path)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    started = time.time()
    while evicted() and time.time() - started < timeout_s:
        time.sleep(2)
    still = evicted()
    if still:
        raise SystemExit(
            f"{len(still)} files are still not available locally after {timeout_s:.0f}s "
            f"(e.g. {still[0]}). Download them in Finder or disable iCloud optimisation."
        )
    print("All dataset files are local.", flush=True)


def build_source_only_dir(dataset_dir: Path) -> tempfile.TemporaryDirectory:
    sources = sorted(dataset_dir.glob("Q*_source_*.md"))
    ensure_local_copies(sources)
    temp = tempfile.TemporaryDirectory(prefix="govrag_conflictbench_all_")
    temp_path = Path(temp.name)
    for source in sources:
        # Copy rather than symlink so a later eviction cannot stall the run.
        shutil.copyfile(source, temp_path / source.name)
    return temp


def load_roles(dataset_dir: Path) -> dict[tuple[str, str], str]:
    """(question_id, filename) -> role from the manifest, for diagnostics only."""
    manifest = dataset_dir / "ground_truth_manifest.csv"
    roles: dict[tuple[str, str], str] = {}
    if not manifest.exists():
        return roles
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roles[(row["question_id"], row["document"])] = row["role"]
    return roles


def build_rag(args, source_dir: Path) -> GovRAGGemini:
    if args.provider == "nvidia":
        from pipeline.providers import LocalEmbedder, OpenAICompatGenerator

        embedder = LocalEmbedder(model_name=args.embed_model)
        generator = None
        if not args.no_llm:
            generator = OpenAICompatGenerator(model=args.generation_model)
        return GovRAGGemini(
            str(source_dir),
            use_llm=not args.no_llm,
            embedder=embedder,
            generator=generator,
        )
    return GovRAGGemini(
        str(source_dir),
        project_id=args.project_id,
        region=args.region,
        use_llm=not args.no_llm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-dir",
        default="../conflictbench_fictional_all",
        help="Flat dataset directory containing Q*_source_*.md and questions.csv",
    )
    parser.add_argument("--provider", choices=["vertex", "nvidia"], default="vertex")
    parser.add_argument("--mode", choices=["govrag", "baseline"], default="govrag")
    parser.add_argument("--generation-model", default=None, help="Override the generation model id")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5", help="Local embedding model (nvidia provider)")
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", "project-79920195-9e86-44ea-8c9"),
    )
    parser.add_argument("--region", default=os.getenv("VERTEX_AI_REGION", "us-central1"))
    parser.add_argument("--output-dir", default="../results_gemini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between questions (rate limits)")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    load_dotenv_file(HERE / ".env")

    dataset_dir = Path(args.dataset_dir).resolve()
    questions_path = dataset_dir / "questions.csv"
    if not questions_path.exists():
        raise SystemExit(f"questions.csv not found: {questions_path}")

    rows = list(csv.DictReader(questions_path.open(newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.provider}_{args.mode}"
    detail_csv = output_dir / f"conflictbench_all_{tag}_details_{stamp}.csv"
    metrics_json = output_dir / f"conflictbench_all_{tag}_metrics_{stamp}.json"
    log_path = output_dir / f"conflictbench_all_{tag}_run_{stamp}.log"

    source_dir_handle = build_source_only_dir(dataset_dir)
    source_dir = Path(source_dir_handle.name)

    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Source docs: {len(list(source_dir.glob('*.md')))}", flush=True)
    print(f"Questions: {len(rows)}", flush=True)
    print(f"Provider: {args.provider}  Mode: {args.mode}", flush=True)
    if args.provider == "vertex":
        print(f"Project: {args.project_id}  Region: {args.region}", flush=True)
    print(f"Details CSV: {detail_csv}", flush=True)
    print(f"Metrics JSON: {metrics_json}", flush=True)
    print(f"Run log: {log_path}", flush=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file):
                rag = build_rag(args, source_dir)

        evaluated = []
        correct_answers = 0
        gold_selected = 0
        gold_in_context = 0
        decisions: Counter[str] = Counter()
        selected_roles: Counter[str] = Counter()
        latencies = []

        for index, row in enumerate(rows, 1):
            question = row["question"]
            qid = row["question_id"]

            query_log = io.StringIO()
            started = time.perf_counter()
            with redirect_stdout(query_log):
                result = rag.query(question, mode=args.mode)
            latency = time.perf_counter() - started
            latencies.append(latency)

            # Ground truth is consulted only after the prediction exists.
            gold_answer = row["gold_answer"]
            gold_document = row["gold_document"]

            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n\n===== {qid} =====\n")
                log_file.write(query_log.getvalue())
                log_file.write("\nRESULT:\n")
                log_file.write(json.dumps(result, default=str, indent=1))
                log_file.write("\n")

            predicted = result.get("answer")
            selected = result.get("source")
            decision = result.get("decision")
            context_files = [c.get("filename") for c in result.get("candidates", [])]
            answer_correct = answer_matches(predicted, gold_answer)
            source_correct = selected == gold_document
            gold_present = gold_document in context_files

            correct_answers += int(answer_correct)
            gold_selected += int(source_correct)
            gold_in_context += int(gold_present)
            decisions[decision or "none"] += 1

            print(
                f"{index:02d}/{len(rows)} {qid}: "
                f"decision={decision} pred={predicted!r} gold={gold_answer!r} "
                f"source={selected!r} gold_doc={gold_document!r} "
                f"answer={'OK' if answer_correct else 'MISS'} "
                f"source={'OK' if source_correct else 'MISS'} "
                f"gold_in_ctx={'Y' if gold_present else 'N'} "
                f"{latency:.1f}s",
                flush=True,
            )

            evaluated.append(
                {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted,
                    "answer_correct": answer_correct,
                    "decision": decision,
                    "gold_document": gold_document,
                    "selected_document": selected,
                    "gold_selected": source_correct,
                    "gold_in_context": gold_present,
                    "context_documents": "|".join(f for f in context_files if f),
                    "conflict_detected": result.get("conflict_detected"),
                    "confidence": result.get("confidence"),
                    "validation_error": bool(result.get("validation_error", False)),
                    "repair_attempts": int(result.get("repair_attempts", 0) or 0),
                    "latency_s": round(latency, 2),
                    "reason": result.get("reason"),
                }
            )

            if args.delay and index < len(rows):
                time.sleep(args.delay)

        roles = load_roles(dataset_dir)
        for item in evaluated:
            sel = item["selected_document"]
            if not sel:
                selected_roles["none"] += 1
            else:
                selected_roles[roles.get((item["question_id"], sel), "unknown")] += 1

        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(evaluated[0].keys()))
            writer.writeheader()
            writer.writerows(evaluated)

        total = len(evaluated)
        metrics = {
            "provider": args.provider,
            "mode": args.mode,
            "generation_model": getattr(getattr(rag, "generator", None), "model", None)
            if args.provider == "nvidia" else ("gemini-2.5-pro" if not args.no_llm else None),
            "embed_model": args.embed_model if args.provider == "nvidia" else "text-embedding-004",
            "total": total,
            "answer_correct": correct_answers,
            "answer_accuracy": round(correct_answers / total * 100, 2),
            "gold_selected": gold_selected,
            "gold_selection_rate": round(gold_selected / total * 100, 2),
            "gold_in_context": gold_in_context,
            "gold_in_context_rate": round(gold_in_context / total * 100, 2),
            "decisions": dict(decisions),
            "validation_errors": sum(1 for item in evaluated if item["validation_error"]),
            "repairs_used": sum(item["repair_attempts"] for item in evaluated),
            "selected_document_roles": dict(selected_roles),
            "mean_latency_s": round(sum(latencies) / total, 2),
            "details_csv": str(detail_csv),
            "run_log": str(log_path),
        }
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print("\nSUMMARY", flush=True)
        print(f"Provider/mode: {args.provider}/{args.mode}", flush=True)
        print(f"Answer accuracy: {correct_answers}/{total} = {metrics['answer_accuracy']:.2f}%", flush=True)
        print(f"Gold source selected: {gold_selected}/{total} = {metrics['gold_selection_rate']:.2f}%", flush=True)
        print(f"Gold in final context: {gold_in_context}/{total} = {metrics['gold_in_context_rate']:.2f}%", flush=True)
        print(f"Decisions: {dict(decisions)}", flush=True)
        print(
            f"Validator rejections (final): {metrics['validation_errors']}  "
            f"repairs used: {metrics['repairs_used']}",
            flush=True,
        )
        print(f"Selected document roles: {dict(selected_roles)}", flush=True)
        print(f"Details CSV: {detail_csv}", flush=True)
        print(f"Metrics JSON: {metrics_json}", flush=True)
        print(f"Run log: {log_path}", flush=True)
    finally:
        source_dir_handle.cleanup()


if __name__ == "__main__":
    main()
