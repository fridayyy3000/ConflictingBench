#!/usr/bin/env python3
"""Evaluate the flat conflictbench_fictional_all folder with GOV-RAG."""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import tempfile
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from pipeline.gov_rag_gemini import GovRAGGemini


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


def build_source_only_dir(dataset_dir: Path) -> tempfile.TemporaryDirectory:
    temp = tempfile.TemporaryDirectory(prefix="govrag_conflictbench_all_")
    temp_path = Path(temp.name)
    for source in sorted(dataset_dir.glob("Q*_source_*.md")):
        target = temp_path / source.name
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copyfile(source, target)
    return temp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="../conflictbench_fictional_all",
        help="Flat dataset directory containing Q*_source_*.md and questions.csv",
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", "project-79920195-9e86-44ea-8c9"),
    )
    parser.add_argument("--region", default=os.getenv("VERTEX_AI_REGION", "us-central1"))
    parser.add_argument("--output-dir", default="../results_gemini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

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
    detail_csv = output_dir / f"conflictbench_all_details_{stamp}.csv"
    log_path = output_dir / f"conflictbench_all_run_{stamp}.log"

    source_dir_handle = build_source_only_dir(dataset_dir)
    source_dir = Path(source_dir_handle.name)

    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Source docs: {len(list(source_dir.glob('*.md')))}", flush=True)
    print(f"Questions: {len(rows)}", flush=True)
    print(f"Project: {args.project_id}", flush=True)
    print(f"Region: {args.region}", flush=True)
    print(f"Details CSV: {detail_csv}", flush=True)
    print(f"Run log: {log_path}", flush=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file):
                rag = GovRAGGemini(
                    str(source_dir),
                    project_id=args.project_id,
                    region=args.region,
                    use_llm=not args.no_llm,
                )

        evaluated = []
        correct_answers = 0
        gold_selected = 0

        for index, row in enumerate(rows, 1):
            question = row["question"]
            qid = row["question_id"]
            gold_answer = row["gold_answer"]
            gold_document = row["gold_document"]

            query_log = io.StringIO()
            with redirect_stdout(query_log):
                result = rag.query(question)

            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n\n===== {qid} =====\n")
                log_file.write(query_log.getvalue())
                log_file.write("\nRESULT:\n")
                log_file.write(str(result))
                log_file.write("\n")

            predicted = result.get("answer")
            selected = result.get("source")
            answer_correct = answer_matches(predicted, gold_answer)
            source_correct = selected == gold_document

            correct_answers += int(answer_correct)
            gold_selected += int(source_correct)

            print(
                f"{index:02d}/{len(rows)} {qid}: "
                f"pred={predicted!r} gold={gold_answer!r} "
                f"source={selected!r} gold_doc={gold_document!r} "
                f"answer={'OK' if answer_correct else 'MISS'} "
                f"source={'OK' if source_correct else 'MISS'}",
                flush=True,
            )

            evaluated.append(
                {
                    "question_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted,
                    "answer_correct": answer_correct,
                    "gold_document": gold_document,
                    "selected_document": selected,
                    "gold_selected": source_correct,
                    "conflict_detected": result.get("conflict_detected"),
                    "confidence": result.get("confidence"),
                    "reason": result.get("reason"),
                }
            )

        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(evaluated[0].keys()))
            writer.writeheader()
            writer.writerows(evaluated)

        total = len(evaluated)
        print("\nSUMMARY", flush=True)
        print(f"Answer accuracy: {correct_answers}/{total} = {correct_answers / total * 100:.2f}%", flush=True)
        print(f"Gold source selected: {gold_selected}/{total} = {gold_selected / total * 100:.2f}%", flush=True)
        print(f"Details CSV: {detail_csv}", flush=True)
        print(f"Run log: {log_path}", flush=True)
    finally:
        source_dir_handle.cleanup()


if __name__ == "__main__":
    main()
