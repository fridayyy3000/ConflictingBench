"""Command line entry point: ``python -m docuinsight ask --docs DIR "question"``."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from . import DEFAULT_POLICY, DEFAULT_RETRIEVAL, build_pipeline, load_folder


def _load_env_file(path: Path) -> None:
    """Minimal KEY=VALUE loader; never prints values."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _print_result(result: dict) -> None:
    print("\n" + "=" * 72)
    print(f"Decision : {result.get('decision')}"
          + ("   (conflict flagged)" if result.get("conflict_detected") else ""))
    print(f"Answer   : {result.get('answer')}")
    shorts = result.get("short_answers") or []
    if shorts:
        print(f"Short    : {', '.join(map(str, shorts))}")
    print(f"Confidence: {result.get('confidence')}")
    if result.get("reason"):
        print(f"Reason   : {result['reason']}")
    cites = result.get("citations") or []
    if cites:
        print("\nCitations:")
        for c in cites:
            treatment = c.get("treatment", "")
            quote = (c.get("quote") or "").strip().replace("\n", " ")
            print(f"  [{treatment:<11}] {c.get('filename')}: \"{quote}\"")
    if result.get("explanation"):
        print(f"\nWhy: {result['explanation']}")
    print("=" * 72)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="docuinsight", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="index a folder of documents and answer one question")
    ask.add_argument("question")
    ask.add_argument("--docs", required=True, help="folder with .md/.txt/.pdf/.docx files (searched recursively)")
    ask.add_argument("--model", default=None, help="chat model name (default: NEMOTRON_MODEL env or provider default)")
    ask.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (default: DOCUINSIGHT_BASE_URL env or NVIDIA)")
    ask.add_argument("--embed-model", default=None, help="sentence-transformers model (default: BAAI/bge-small-en-v1.5)")
    ask.add_argument("--retrieval", choices=["hybrid", "dense"], default=DEFAULT_RETRIEVAL,
                     help="first-stage retrieval; 'dense' is the ablation")
    ask.add_argument("--policy", choices=["answer_and_flag", "strict"], default=DEFAULT_POLICY,
                     help="resolution policy; 'strict' is the refuse-on-conflict ablation")
    ask.add_argument("--env-file", default=None, help="KEY=VALUE file to load into the environment (e.g. an .env with NVIDIA_API_KEY)")
    ask.add_argument("--json", action="store_true", help="print the full result as JSON instead of a summary")
    args = parser.parse_args(argv)

    if args.env_file:
        _load_env_file(Path(args.env_file))

    # The core prints its progress to stdout; in --json mode keep stdout clean.
    sink = sys.stderr if args.json else sys.stdout
    with contextlib.redirect_stdout(sink):
        documents = load_folder(args.docs)
        pipeline = build_pipeline(documents, model=args.model, base_url=args.base_url, embed_model=args.embed_model)
        result = pipeline.query(args.question, retrieval=args.retrieval, policy=args.policy)

    if args.json:
        slim = {k: v for k, v in result.items() if k != "candidates"}
        json.dump(slim, sys.stdout, indent=2, default=str)
        print()
    else:
        _print_result(result)
    return 0
