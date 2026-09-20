#!/usr/bin/env python3
"""Verify durable corpus index behavior without calling Vertex/Gemini."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np

import main
import pipeline.gov_rag_gemini as gov_rag
from pipeline.corpus_store import CorpusStore


class FakeEmbedder:
    encode_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vec = np.zeros(8, dtype=float)
        for idx, byte in enumerate(text.encode("utf-8")):
            vec[idx % len(vec)] += float(byte)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        FakeEmbedder.encode_calls += 1
        return [self._vector(f"{task_type}:{text}") for text in texts]


def reset_memory_cache() -> None:
    main.rag_instances.clear()
    main.rag_instance_metadata.clear()
    main.corpus_locks.clear()


def add_doc(store: CorpusStore, corpus_id: str, filename: str, text: str) -> str:
    document_id = str(uuid.uuid4())
    normalized_filename = f"{document_id}_{filename}"
    store.add_documents_bulk(
        corpus_id,
        [{
            "document_id": document_id,
            "original_filename": filename,
            "normalized_filename": normalized_filename,
            "file_type": ".md",
            "size_bytes": len(text.encode("utf-8")),
            "normalized_text": text,
        }],
    )
    return document_id


def ensure_without_chunk_reembed(corpus_id: str, request: main.QueryRequest):
    FakeEmbedder.encode_calls = 0
    rag, cache_meta = main.ensure_corpus_rag(corpus_id, request, force_reindex=False)
    assert FakeEmbedder.encode_calls == 0, "persisted index load re-embedded corpus chunks"
    return rag, cache_meta


def main_verify() -> None:
    gov_rag.GeminiEmbedder = FakeEmbedder

    with tempfile.TemporaryDirectory() as tmp:
        main.corpus_store = CorpusStore(base_dir=tmp, bucket_name=None)
        reset_memory_cache()

        corpus_id = str(uuid.uuid4())
        main.corpus_store.create_corpus(corpus_id, "Persisted Index Verification")
        first_doc_id = add_doc(
            main.corpus_store,
            corpus_id,
            "policy_a.md",
            "Status: final\nDocument Type: policy\nThe retention period is 30 days.",
        )
        add_doc(
            main.corpus_store,
            corpus_id,
            "policy_b.md",
            "Status: draft\nDocument Type: memo\nThe draft mentions a 60 day period.",
        )

        request = main.QueryRequest(
            question="What is the retention period?",
            project_id="test-project",
            region="us-central1",
            use_llm=False,
            top_k=4,
        )

        FakeEmbedder.encode_calls = 0
        main.ensure_corpus_rag(corpus_id, request, force_reindex=True)
        assert FakeEmbedder.encode_calls > 0, "initial index did not embed chunks"

        persisted = main.corpus_store.load_corpus_index(corpus_id)
        assert persisted["metadata"]["corpus_id"] == corpus_id
        assert Path(tmp, "corpora", corpus_id, "index", "chunks.parquet").exists()
        assert Path(tmp, "corpora", corpus_id, "index", "metadata.json").exists()

        reset_memory_cache()
        ensure_without_chunk_reembed(corpus_id, request)

        response = main.run_corpus_query(corpus_id, request)
        assert response.corpus_id == corpus_id
        assert response.top_sources, "query returned no sources"

        add_doc(
            main.corpus_store,
            corpus_id,
            "policy_c.md",
            "Status: final\nDocument Type: statute\nThe controlling retention period is 45 days.",
        )
        reset_memory_cache()
        FakeEmbedder.encode_calls = 0
        main.ensure_corpus_rag(corpus_id, request, force_reindex=False)
        assert FakeEmbedder.encode_calls > 0, "upload invalidation did not rebuild the index"

        reset_memory_cache()
        ensure_without_chunk_reembed(corpus_id, request)

        main.corpus_store.delete_document(corpus_id, first_doc_id)
        reset_memory_cache()
        FakeEmbedder.encode_calls = 0
        main.ensure_corpus_rag(corpus_id, request, force_reindex=False)
        assert FakeEmbedder.encode_calls > 0, "delete invalidation did not rebuild the index"

        reset_memory_cache()
        ensure_without_chunk_reembed(corpus_id, request)

    print("PASS: persisted corpus index rebuild/load/invalidation verified")


if __name__ == "__main__":
    main_verify()
