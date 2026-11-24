"""Alternative embedding and generation providers for GOV-RAG.

These let the pipeline run without Google Cloud:

- ``LocalEmbedder``: sentence-transformers model run on this machine
  (default ``BAAI/bge-small-en-v1.5``, which is what the original
  ``code/gov_rag.py`` used). No network calls after the model is cached.
- ``OpenAICompatGenerator``: any OpenAI-compatible chat endpoint. Used here
  for NVIDIA's hosted Nemotron models (``https://integrate.api.nvidia.com/v1``)
  but works for any endpoint that speaks the same protocol.

Both expose the same interface as ``GeminiEmbedder`` / ``GeminiGenerator`` so
``GovRAGGemini`` can take them as drop-in replacements.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from typing import List, Optional

import numpy as np

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge models are trained with this instruction prefixed to *queries* only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LocalEmbedder:
    """sentence-transformers embedder with the GeminiEmbedder interface."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_EMBED_MODEL, batch_size: int = 32,
                 device: Optional[str] = None, cache_path: Optional[str] = None):
        # If the model is already in the local Hugging Face cache, skip the
        # online freshness check: it can stall indefinitely on some networks.
        cache_dir = os.path.expanduser(
            os.getenv("HF_HUB_CACHE") or os.path.join(os.getenv("HF_HOME", "~/.cache/huggingface"), "hub")
        )
        if os.path.isdir(os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        # bge-small is fast enough on CPU; defaulting to CPU keeps evaluation
        # runs off the GPU (EMBED_DEVICE=mps|cuda|cpu overrides).
        self.device = device or os.getenv("EMBED_DEVICE", "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.use_vertex = False
        self.cache = EmbeddingCache.from_env(model_name, cache_path)
        self.cache_hits = 0
        self.cache_misses = 0
        print("✓ Using local sentence-transformers embeddings")
        print(f"  Model: {model_name}  Device: {self.device}")
        if self.cache is not None:
            print(f"  Embedding cache: {self.cache.path} ({self.cache.count()} vectors)")

    def encode(self, texts, task_type: str = "RETRIEVAL_DOCUMENT"):
        if isinstance(texts, str):
            texts = [texts]
        if task_type == "RETRIEVAL_QUERY" and "bge" in self.model_name.lower():
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        texts = list(texts)
        if self.cache is None:
            return self._encode_uncached(texts)
        cached = self.cache.get_many(texts)
        missing = [i for i, v in enumerate(cached) if v is None]
        self.cache_hits += len(texts) - len(missing)
        self.cache_misses += len(missing)
        if missing:
            fresh = self._encode_uncached([texts[i] for i in missing])
            self.cache.put_many([texts[i] for i in missing], fresh)
            for i, vec in zip(missing, fresh):
                cached[i] = vec
        return np.asarray(np.stack(cached), dtype=np.float64)

    def _encode_uncached(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float64)


class EmbeddingCache:
    """Persistent text -> vector store (SQLite) shared by every run.

    Keyed by SHA-256 of ``model_name`` and the exact text handed to the model
    (query prefix included), so a hit is byte-identical to re-encoding.
    Disable with ``EMBED_CACHE=0``; relocate with ``EMBED_CACHE_PATH``.
    Default location: ``<repo>/data/embeddings/<model-slug>.sqlite``.
    """

    def __init__(self, path: str, model_name: str):
        self.path = path
        self.model_name = model_name
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=60.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self._conn.commit()

    @classmethod
    def from_env(cls, model_name: str, path: Optional[str] = None) -> Optional["EmbeddingCache"]:
        if os.getenv("EMBED_CACHE", "1").lower() in ("0", "false", "off"):
            return None
        if path is None:
            path = os.getenv("EMBED_CACHE_PATH")
        if path is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-")
            path = os.path.join(repo_root, "data", "embeddings", f"{slug}.sqlite")
        return cls(os.path.expanduser(path), model_name)

    def _key(self, text: str) -> str:
        return hashlib.sha256((self.model_name + "\x1f" + text).encode("utf-8")).hexdigest()

    def get_many(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        keys = [self._key(t) for t in texts]
        found = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                rows = self._conn.execute(
                    f"SELECT key, dim, vec FROM embeddings WHERE key IN ({','.join('?' * len(chunk))})", chunk
                ).fetchall()
                for key, dim, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32, count=dim).astype(np.float64)
        return [found.get(k) for k in keys]

    def put_many(self, texts: List[str], vectors: np.ndarray) -> None:
        rows = []
        for text, vec in zip(texts, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            rows.append((self._key(text), int(arr.shape[0]), arr.tobytes()))
        with self._lock:
            self._conn.executemany("INSERT OR IGNORE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)", rows)
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


class OpenAICompatGenerator:
    """Chat-completions generator with the GeminiGenerator interface.

    ``generate(question, candidates, context=None, baseline=False)`` returns a
    validated decision dict exactly like ``GeminiGenerator.generate``. Prompt
    construction, JSON parsing and citation validation are shared with the
    Gemini path via ``pipeline.gov_rag_gemini`` so the two providers see the
    same evidence formatting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = NVIDIA_BASE_URL,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        max_retries: int = 10,
        enable_thinking: Optional[bool] = None,
    ):
        from openai import OpenAI

        key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No API key for the OpenAI-compatible generator "
                "(set NVIDIA_API_KEY in the environment or corvic/.env)."
            )
        self.model = model or os.getenv("NEMOTRON_MODEL") or DEFAULT_NVIDIA_MODEL
        self.temperature = temperature
        # Reasoning models spend part of the budget on thinking before the
        # JSON answer, so the budget is larger than the Gemini path's 4096.
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        if enable_thinking is None:
            enable_thinking = os.getenv("NEMOTRON_THINKING", "1").lower() not in ("0", "false", "off")
        self.enable_thinking = enable_thinking
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
        self.use_vertex = False
        print("✓ Using OpenAI-compatible generation endpoint")
        print(f"  Base URL: {base_url}")
        print(f"  Model: {self.model}")
        print(f"  Thinking: {'on' if self.enable_thinking else 'off'}")

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        delay = 2.0
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {}
                if not self.enable_thinking:
                    # Nemotron 3.x chat templates accept this toggle.
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                if choice.finish_reason not in (None, "stop"):
                    print(f"  WARNING: completion finish_reason={choice.finish_reason} ({len(content)} chars)")
                # Reasoning models may emit a <think> block before the answer.
                content = _THINK_BLOCK.sub("", content).strip()
                if not content:
                    raise RuntimeError(
                        f"empty completion (finish_reason={choice.finish_reason}); "
                        "the reasoning budget was probably exhausted before the answer"
                    )
                return content
            except Exception as exc:  # noqa: BLE001 - retry on transient API errors
                last_error = exc
                status = getattr(exc, "status_code", None)
                retryable = status in (408, 409, 429, 500, 502, 503, 504) or status is None
                if not retryable or attempt == self.max_retries - 1:
                    raise
                print(f"  generation attempt {attempt + 1} failed ({status or type(exc).__name__}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"generation failed after retries: {last_error}")

    def closed_book(self, question, context=None):
        """Direct few-shot CoT answer with no passages (``closedbook`` mode).

        Returns a dict with the ``generate`` result keys that make sense
        without evidence; citations are always empty and nothing is validated
        against passages, but malformed JSON gets one repair attempt.
        """
        from pipeline import gov_rag_gemini as core

        context = {k: v for k, v in (context or {}).items() if v is not None and str(v).strip()}
        user_prompt = f"Question: {question}"
        if context:
            user_prompt += "\nCase context: " + " ".join(f"{k}: {v}" for k, v in context.items())
        repairs = 0
        raw = self._chat(core.CLOSED_BOOK_SYSTEM_PROMPT, user_prompt)
        parsed = core.parse_json_response(raw)
        if not isinstance(parsed, dict) or "answer" not in parsed:
            repairs = 1
            raw = self._chat(core.CLOSED_BOOK_SYSTEM_PROMPT,
                             user_prompt + "\n\nReturn only the JSON object described above.")
            parsed = core.parse_json_response(raw)
        if not isinstance(parsed, dict) or "answer" not in parsed:
            failed = core.generation_failure("Closed-book generation returned no parseable JSON answer.")
            failed["repair_attempts"] = repairs
            return failed
        decision = parsed.get("decision", "answered")
        if decision not in ("answered", "insufficient_evidence"):
            decision = "answered"
        short = parsed.get("short_answers") or []
        if not isinstance(short, list):
            short = [short]
        short = [str(x).strip() for x in short if str(x).strip()]
        answer = str(parsed.get("answer") or "").strip()
        if decision == "insufficient_evidence" or not answer:
            decision = "insufficient_evidence"
            short = []
            answer = answer or "I do not know."
        confidence = parsed.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        reasoning = str(parsed.get("reasoning") or "")
        return {
            "decision": decision,
            "answer": answer,
            "short_answers": short,
            "citations": [],
            "reason": reasoning,
            "explanation": reasoning,
            "selected_source": None,
            "conflict_detected": False,
            "confidence": confidence,
            "validation_error": False,
            "repair_attempts": repairs,
            "dropped_citations": 0,
        }

    def generate(self, question, candidates, context=None, baseline: bool = False, policy: str = "strict"):
        # Imported lazily to avoid a circular import at module load time.
        from pipeline import gov_rag_gemini as core

        if baseline:
            system_prompt = core.BASELINE_SYSTEM_PROMPT
        elif policy == "answer_and_flag":
            system_prompt = core.SYSTEM_PROMPT_ANSWER_AND_FLAG
        else:
            system_prompt = core.SYSTEM_PROMPT
        user_prompt = core.build_user_prompt(
            question, candidates, context=context, include_signals=not baseline,
            corroboration=(not baseline and policy == "answer_and_flag"),
        )
        try:
            return core.run_generation(self._chat, system_prompt, user_prompt, candidates)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: generation failed: {exc}")
            return core.generation_failure("Generation failed; no answer was selected.")
