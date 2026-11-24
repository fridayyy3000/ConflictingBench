"""Rankify toolkit baseline (ACL 2026 system demonstration), driven in a subprocess pool.

See ``rankify_worker.py`` for the configuration. This side owns a pool of
worker processes (one per harness worker thread), hands each question with its
documents to a free worker and maps the free-text reply onto the result dict
the evaluation harness expects (same shape as the LlamaIndex baseline).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from pipeline.providers import NVIDIA_BASE_URL, DEFAULT_NVIDIA_MODEL

WORKER_PY = Path(os.environ.get("RANKIFY_PYTHON", str(Path.home() / ".venvs" / "rankify" / "bin" / "python")))
WORKER_SCRIPT = Path(__file__).with_name("rankify_worker.py")


class _Worker:
    def __init__(self, cmd, env):
        self.cmd, self.env = cmd, env
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.start()

    def start(self):
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                                     text=True, bufsize=1, env=self.env)
        ready = self.proc.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            raise RuntimeError("Rankify worker failed to start (see stderr)")

    def ask(self, req: dict, timeout: float) -> dict:
        with self.lock:
            if self.proc.poll() is not None:
                self.start()
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            result: dict = {}

            def _read():
                line = self.proc.stdout.readline()
                result["line"] = line

            t = threading.Thread(target=_read, daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive() or not result.get("line"):
                self.proc.kill()
                self.start()
                return {"id": req["id"], "error": f"worker timeout after {timeout:.0f}s"}
            return json.loads(result["line"])


class RankifyBaseline:
    def __init__(self, model: Optional[str] = None, workers: int = 1, base_url: str = NVIDIA_BASE_URL,
                 reranker_method: str = "monot5", reranker_model: str = "monot5-base-msmarco",
                 generator_method: str = "basic-rag", max_tokens: int = 4096, top_k: int = 0,
                 item_timeout: float = 900.0):
        if not WORKER_PY.exists():
            raise RuntimeError(f"Rankify venv python not found at {WORKER_PY} (set RANKIFY_PYTHON)")
        key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key for the Rankify baseline (set NVIDIA_API_KEY in corvic/.env).")
        self.model = model or os.getenv("NEMOTRON_MODEL") or DEFAULT_NVIDIA_MODEL
        self.top_k = top_k
        self.item_timeout = item_timeout
        self.config = {"reranker": f"{reranker_method}/{reranker_model}", "generator": generator_method,
                       "max_tokens": max_tokens, "temperature": 0.0, "top_k": top_k or "all"}
        cmd = [str(WORKER_PY), "-u", str(WORKER_SCRIPT), "--model", self.model, "--base-url", base_url,
               "--reranker-method", reranker_method, "--reranker-model", reranker_model,
               "--generator-method", generator_method, "--max-tokens", str(max_tokens), "--top-k", str(top_k)]
        env = dict(os.environ, NVIDIA_API_KEY=key, TOKENIZERS_PARALLELISM="false")
        print("✓ Rankify baseline (subprocess pool)")
        print(f"  {self.config}  model={self.model}  workers={workers}")
        self.pool: "queue.Queue[_Worker]" = queue.Queue()
        for _ in range(max(1, workers)):
            self.pool.put(_Worker(cmd, env))

    def query(self, question: str, documents) -> dict:
        started = time.perf_counter()
        req = {"id": f"{time.time_ns()}", "question": question,
               "passages": [{"id": d.filename, "title": "", "text": d.text} for d in documents]}
        w = self.pool.get()
        try:
            reply = w.ask(req, self.item_timeout)
        finally:
            self.pool.put(w)
        by_id = {d.filename: d for d in documents}
        candidates = [
            {"filename": r["id"], "chunk_id": r["id"], "semantic_score": r.get("score"),
             "final_score": r.get("score"), "text": getattr(by_id.get(r["id"]), "text", "")}
            for r in reply.get("ranked", [])
        ]
        error = reply.get("error")
        return {
            "question": question,
            "decision": "free_text",
            "answer": "" if error else (reply.get("answer") or ""),
            "short_answers": [],
            "citations": [],
            "candidates": candidates,
            "source": None,
            "selected_source": None,
            # Harness sentinel: not checkpointed, so a resumed run retries the item.
            "reason": "Generation failed; no answer was selected." if error else None,
            "explanation": error,  # keep the worker's error text for diagnosis
            "confidence": None,
            "validation_error": bool(error),
            "repair_attempts": 0,
            "dropped_citations": 0,
            "retrieval_mode": f"rankify_{self.config['reranker'].split('/')[0]}_{self.config['generator']}",
            "_latency_s": time.perf_counter() - started,
        }
