"""Rankify worker process (runs under the Rankify virtualenv, not the project env).

Rankify pins old transformers / sentence-transformers / httpx / openai versions
that conflict with the project environment, so it lives in its own venv at
~/.venvs/rankify and the evaluation harness talks to this script over a
line-oriented JSON protocol on stdin/stdout. One request per line:

    {"id": "...", "question": "...", "passages": [{"id": "...", "title": "...", "text": "..."}]}

One reply per line:

    {"id": "...", "answer": "...", "ranked": [{"id": "...", "score": 0.93}, ...], "latency_s": 1.2}

Configuration mirrors Rankify's README headline example: the monoT5 MS MARCO
reranker followed by the ``basic-rag`` generator with the library's default
prompt (single user message, no system prompt). All reranked passages are
passed to the generator, as basic-rag does by default. Two deliberate changes
from library defaults, both reported in the paper: temperature 0 instead of
0.7 for determinism, and a larger max_tokens because the generator is a
reasoning model and 128 tokens would be spent before the answer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types

# Rankify imports vllm unconditionally; it is Linux-only. Stub it.
class _StubModule(types.ModuleType):
    def __getattr__(self, name):  # any class/function Rankify asks for
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


sys.modules.setdefault("vllm", _StubModule("vllm"))

# Keep stdout for the protocol only; library chatter goes to stderr.
_proto_out = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr

from rankify.dataset.dataset import Answer, Context, Document, Question  # noqa: E402
from rankify.generator.generator import Generator  # noqa: E402
from rankify.models.reranking import Reranking  # noqa: E402

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker-method", default="monot5")
    ap.add_argument("--reranker-model", default="monot5-base-msmarco")
    ap.add_argument("--generator-method", default="basic-rag")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 = pass every reranked passage (Rankify default)")
    ap.add_argument("--request-timeout", type=float, default=300.0)
    a = ap.parse_args()

    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("NVIDIA_API_KEY not set for the Rankify worker")

    reranker = Reranking(method=a.reranker_method, model_name=a.reranker_model)
    gen = Generator(method=a.generator_method, model_name=a.model, backend="openai",
                    api_keys=[key], base_url=a.base_url)
    # Make sure the OpenAI client really points at the requested endpoint
    # (PyPI 0.1.4 dropped base_url in the model factory; main forwards it) and
    # lift Rankify's hard-coded 30 s request timeout, which a reasoning model
    # exceeds and which its infinite retry loop would otherwise spin on.
    rk_client = gen.rag_method.model.client          # rankify OpenaiClient
    client = rk_client.client                         # openai.OpenAI
    client.base_url = a.base_url

    # Rankify's OpenaiClient.chat() retries forever with a 30 s timeout and
    # swallows the error text into stdout. Replace it with a bounded, logged
    # retry so a persistent failure surfaces instead of hanging the run.
    def _bounded_chat(*args, return_text=False, reduce_length=False, **kwargs):
        kwargs.pop("timeout", None)
        delay, last = 2.0, None
        for attempt in range(10):
            try:
                completion = client.chat.completions.create(*args, **kwargs, timeout=a.request_timeout)
                choice = completion.choices[0]
                if choice.finish_reason not in (None, "stop"):
                    print(f"  rankify completion finish_reason={choice.finish_reason}", file=sys.stderr, flush=True)
                return choice.message.content if return_text else completion
            except Exception as exc:  # noqa: BLE001
                last = exc
                status = getattr(exc, "status_code", None)
                print(f"  rankify generation attempt {attempt + 1} failed ({status or type(exc).__name__}): "
                      f"{str(exc)[:300]}", file=sys.stderr, flush=True)
                if status is not None and status not in (408, 409, 429, 500, 502, 503, 504):
                    break
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"generation failed: {last}")

    rk_client.chat = _bounded_chat
    print(f"rankify worker ready: reranker={a.reranker_method}/{a.reranker_model} "
          f"generator={a.generator_method} model={a.model} base_url={client.base_url}", file=sys.stderr, flush=True)
    _proto_out.write(json.dumps({"ready": True}) + "\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        started = time.perf_counter()
        try:
            contexts = [Context(id=p["id"], title=p.get("title") or "", text=p["text"], score=None, has_answer=None)
                        for p in req["passages"] if (p.get("text") or "").strip()]
            doc = Document(question=Question(req["question"]), answers=Answer([]), contexts=contexts, id=req["id"])
            reranker.rank([doc])
            ranked = doc.reorder_contexts or doc.contexts
            if a.top_k > 0:
                ranked = ranked[: a.top_k]
            doc.contexts = ranked  # generators read .contexts, never .reorder_contexts
            out = gen.generate([doc], max_tokens=a.max_tokens, temperature=a.temperature)
            answer = out[0] if isinstance(out, list) else str(out)
            answer = _THINK_BLOCK.sub("", answer or "").strip()
            reply = {
                "id": req["id"],
                "answer": answer,
                "ranked": [{"id": c.id, "score": (float(c.score) if c.score is not None else None)} for c in ranked],
                "latency_s": time.perf_counter() - started,
            }
        except Exception as exc:  # noqa: BLE001 - report to the parent, keep serving
            reply = {"id": req.get("id"), "error": f"{type(exc).__name__}: {exc}",
                     "latency_s": time.perf_counter() - started}
        _proto_out.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
