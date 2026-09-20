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
    calls_by_task = {}
    texts_by_task = {}

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def reset_counts(cls) -> None:
        cls.encode_calls = 0
        cls.calls_by_task = {}
        cls.texts_by_task = {}

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vec = np.zeros(8, dtype=float)
        for idx, byte in enumerate(text.encode("utf-8")):
            vec[idx % len(vec)] += float(byte)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        FakeEmbedder.encode_calls += 1
        FakeEmbedder.calls_by_task[task_type] = FakeEmbedder.calls_by_task.get(task_type, 0) + 1
        FakeEmbedder.texts_by_task[task_type] = FakeEmbedder.texts_by_task.get(task_type, 0) + len(texts)
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
    reset_memory_cache()
    FakeEmbedder.reset_counts()
    rag, cache_meta = main.ensure_corpus_rag(corpus_id, request, force_reindex=False)
    assert FakeEmbedder.encode_calls == 0, "persisted index load re-embedded corpus chunks"
    return rag, cache_meta


def assert_query_uses_no_document_embedding(corpus_id: str, request: main.QueryRequest) -> None:
    reset_memory_cache()
    FakeEmbedder.reset_counts()
    response = main.run_corpus_query(corpus_id, request)
    assert response.corpus_id == corpus_id
    assert response.top_sources, "query returned no sources"
    assert FakeEmbedder.calls_by_task.get("RETRIEVAL_QUERY", 0) == 1
    assert FakeEmbedder.calls_by_task.get("RETRIEVAL_DOCUMENT", 0) == 0, (
        "query-time reranking performed RETRIEVAL_DOCUMENT embedding calls"
    )


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

        FakeEmbedder.reset_counts()
        main.ensure_corpus_rag(corpus_id, request, force_reindex=True)
        assert FakeEmbedder.calls_by_task.get("RETRIEVAL_DOCUMENT", 0) == 2, (
            "initial index should embed chunks and scope texts"
        )

        persisted = main.corpus_store.load_corpus_index(corpus_id)
        assert persisted["metadata"]["corpus_id"] == corpus_id
        assert persisted["metadata"]["index_format_version"] == main.PERSISTED_INDEX_FORMAT_VERSION
        assert Path(tmp, "corpora", corpus_id, "index", "chunks.parquet").exists()
        assert Path(tmp, "corpora", corpus_id, "index", "metadata.json").exists()
        assert persisted["chunks"], "persisted index contains no chunk rows"
        assert all(row.get("embedding") for row in persisted["chunks"])
        assert all(row.get("scope_embedding") for row in persisted["chunks"])
        assert all(row.get("scope_text") for row in persisted["chunks"])

        ensure_without_chunk_reembed(corpus_id, request)
        assert_query_uses_no_document_embedding(corpus_id, request)

        add_doc(
            main.corpus_store,
            corpus_id,
            "policy_c.md",
            "Status: final\nDocument Type: statute\nThe controlling retention period is 45 days.",
        )
        reset_memory_cache()
        FakeEmbedder.reset_counts()
        main.ensure_corpus_rag(corpus_id, request, force_reindex=False)
        assert FakeEmbedder.calls_by_task.get("RETRIEVAL_DOCUMENT", 0) == 2, (
            "upload invalidation did not rebuild chunk and scope embeddings"
        )

        ensure_without_chunk_reembed(corpus_id, request)
        assert_query_uses_no_document_embedding(corpus_id, request)

        main.corpus_store.delete_document(corpus_id, first_doc_id)
        reset_memory_cache()
        FakeEmbedder.reset_counts()
        main.ensure_corpus_rag(corpus_id, request, force_reindex=False)
        assert FakeEmbedder.calls_by_task.get("RETRIEVAL_DOCUMENT", 0) == 2, (
            "delete invalidation did not rebuild chunk and scope embeddings"
        )

        ensure_without_chunk_reembed(corpus_id, request)
        assert_query_uses_no_document_embedding(corpus_id, request)

    print("PASS: persisted corpus index rebuild/load/invalidation verified")


if __name__ == "__main__":
    main_verify()
