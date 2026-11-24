"""LlamaIndex default-pipeline baseline (an existing, widely deployed RAG system).

Reproduces what a developer gets from the LlamaIndex quickstart with no tuning:
``VectorStoreIndex.from_documents(docs).as_query_engine(similarity_top_k=k)``
with the library's default ``SentenceSplitter`` (1024 tokens, 200 overlap), the
default ``compact`` response synthesiser and its stock QA / refine prompts. Only
the two swappable components are ours: the embedding model is the same local
bge-small encoder (and SQLite cache) used by every other row of the table, and
the LLM is the same NVIDIA-hosted generator reached through LlamaIndex's
OpenAI-like client, so the comparison isolates the retrieval + prompting stack.

The system returns free text: no decision label, no short-answer list and no
citations. ``query`` maps that onto the result dict the evaluation harness
expects so the same scorer and LLM judge run unchanged. Retrieved nodes are
reported as ``candidates`` (context) and not as citations, because the
default pipeline does not cite.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core import Document as LIDocument
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.llms.openai_like import OpenAILike

from pipeline.providers import NVIDIA_BASE_URL, DEFAULT_NVIDIA_MODEL

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
DEFAULT_TOP_K = 2  # LlamaIndex's own default for similarity_top_k


class SharedLocalEmbedding(BaseEmbedding):
    """LlamaIndex embedding backed by the harness's LocalEmbedder (cache included)."""

    _inner: Any = PrivateAttr()

    def __init__(self, inner, **kwargs):
        super().__init__(model_name=getattr(inner, "model_name", "local"), embed_batch_size=32, **kwargs)
        self._inner = inner

    @classmethod
    def class_name(cls) -> str:
        return "SharedLocalEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._inner.encode([query], task_type="RETRIEVAL_QUERY")[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._inner.encode([text], task_type="RETRIEVAL_DOCUMENT")[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._inner.encode(list(texts), task_type="RETRIEVAL_DOCUMENT")]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


class LlamaIndexBaseline:
    def __init__(
        self,
        embedder,
        model: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        base_url: str = NVIDIA_BASE_URL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        enable_thinking: Optional[bool] = None,
    ):
        key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("No API key for the LlamaIndex baseline (set NVIDIA_API_KEY in corvic/.env).")
        self.model = model or os.getenv("NEMOTRON_MODEL") or DEFAULT_NVIDIA_MODEL
        self.top_k = top_k
        if enable_thinking is None:
            enable_thinking = os.getenv("NEMOTRON_THINKING", "1").lower() not in ("0", "false", "off")
        extra: dict = {}
        if not enable_thinking:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        self.llm = OpenAILike(
            model=self.model,
            api_base=base_url,
            api_key=key,
            is_chat_model=True,
            is_function_calling_model=False,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=128_000,
            max_retries=10,
            timeout=300.0,
            additional_kwargs=extra,
        )
        self.embed_model = SharedLocalEmbedding(embedder)
        # Keep the globals pointed at our components so nothing falls back to OpenAI.
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        print("✓ LlamaIndex default pipeline baseline")
        print(f"  LLM: {self.model} via {base_url} (thinking {'on' if enable_thinking else 'off'})")
        print(f"  similarity_top_k={top_k}; node parser / synthesiser / prompts = LlamaIndex defaults")

    def query(self, question: str, documents) -> dict:
        """documents: iterable with ``.filename`` and ``.text`` (pipeline Document)."""
        started = time.perf_counter()
        li_docs = [
            LIDocument(text=d.text, metadata={"filename": d.filename}, excluded_llm_metadata_keys=["filename"],
                       excluded_embed_metadata_keys=["filename"])
            for d in documents if (d.text or "").strip()
        ]
        index = VectorStoreIndex.from_documents(li_docs, embed_model=self.embed_model, show_progress=False)
        engine = index.as_query_engine(llm=self.llm, similarity_top_k=self.top_k)
        # The OpenAI client inside OpenAILike retries 429s only with a short
        # (<=8 s) backoff, which is exhausted quickly when the endpoint is
        # throttled. Wrap it in the same patient schedule the other providers use.
        response = None
        for attempt in range(10):
            try:
                response = engine.query(question)
                break
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                retryable = status in (429, 500, 502, 503, 504) or "429" in str(exc)[:80] or "timeout" in str(exc).lower()
                if not retryable or attempt == 9:
                    raise
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
        answer = _THINK_BLOCK.sub("", str(response) or "").strip()
        candidates = [
            {
                "filename": n.node.metadata.get("filename"),
                "chunk_id": n.node.node_id,
                "semantic_score": float(n.score) if n.score is not None else None,
                "final_score": float(n.score) if n.score is not None else None,
                "text": n.node.get_content(),
            }
            for n in (response.source_nodes or [])
        ]
        return {
            "question": question,
            "decision": "free_text",  # system has no decision schema
            "answer": answer,
            "short_answers": [],
            "citations": [],
            "candidates": candidates,
            "source": None,
            "selected_source": None,
            "reason": None,
            "explanation": None,
            "confidence": None,
            "validation_error": False,
            "repair_attempts": 0,
            "dropped_citations": 0,
            "retrieval_mode": f"llamaindex_default_k{self.top_k}",
            "_latency_s": time.perf_counter() - started,
        }
