"""Offline tests for inspectable evidence and structured decisions."""

from __future__ import annotations

import numpy as np

import corvic.main as app
from corvic.pipeline.chunker import chunk_text_with_locations
from corvic.pipeline import gov_rag_gemini as rag_module
from corvic.pipeline.corpus_store import CorpusStore
from corvic.pipeline.gov_rag_gemini import (
    Candidate,
    Chunk,
    Document,
    DocumentMeta,
    GeminiGenerator,
    GovRAGGemini,
    build_chunks,
    detect_status,
    diversify_conflicts,
    validate_generation_result,
)


def make_candidate(
    chunk_id="doc-a:chunk:0",
    text="Employees may claim 30 days, except during approved leave.",
    filename="doc-a.md",
    document_id="doc-a",
    chunk_index=0,
    page_start=None,
    section=None,
    claim="30 days",
    claim_sentence="Employees may claim 30 days, except during approved leave.",
):
    return Candidate(
        filename=filename,
        text=text,
        semantic_score=0.8,
        authority_score=0.5,
        scope_score=0.7,
        claim=claim,
        claim_sentence=claim_sentence,
        status="unknown",
        doc_type="unknown",
        final_score=0.7,
        chunk_index=chunk_index,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_start,
        section=section,
    )


def test_question_only_api_request_remains_compatible(monkeypatch):
    request = app.QueryRequest.model_validate({"question": "What is the limit?"})
    candidate = make_candidate()
    generated = validate_generation_result(
        {
            "decision": "answered",
            "answer": "30 days, except during approved leave.",
            "citations": [{
                "chunk_id": candidate.chunk_id,
                "quote": "30 days, except during approved leave.",
                "treatment": "governing",
            }],
            "explanation": "The passage states the limit and its exception.",
        },
        [candidate],
    )

    class FakeRag:
        def query(self, question, context=None):
            self.received_context = context
            return {**generated, "source": "doc-a.md", "candidates": [candidate.__dict__]}

    fake_rag = FakeRag()
    monkeypatch.setattr(app, "get_demo_rag_instance", lambda unused_request: fake_rag)
    response = app.run_demo_query(request)
    assert response.question == "What is the limit?"
    assert response.selected_source == "doc-a.md"
    assert response.decision == "answered"
    assert response.citations[0].chunk_id == candidate.chunk_id
    assert fake_rag.received_context is None

    contextual_request = app.QueryRequest.model_validate({
        "question": "What is the limit?",
        "context": {"relevant_date": "2026-10-31", "location": "Ontario"},
    })
    app.run_demo_query(contextual_request)
    assert fake_rag.received_context == {"relevant_date": "2026-10-31", "location": "Ontario"}


def test_generation_citations_are_multiple_and_must_match_retrieved_chunks():
    first = make_candidate(chunk_id="doc-a:chunk:0")
    second = make_candidate(
        chunk_id="doc-a:chunk:1",
        text="Employees may claim 30 days, except during a leave of absence.",
        chunk_index=1,
        claim_sentence="Employees may claim 30 days, except during a leave of absence.",
    )
    valid = validate_generation_result(
        {
            "decision": "answered",
            "answer": "30 days, subject to the stated leave exception.",
            "citations": [
                {"chunk_id": first.chunk_id, "quote": "30 days, except during approved leave.", "treatment": "supporting"},
                {"chunk_id": second.chunk_id, "quote": "except during a leave of absence.", "treatment": "supporting"},
            ],
            "explanation": "Both relevant passages support the answer.",
        },
        [first, second],
    )
    assert len(valid["citations"]) == 2
    assert valid["citations"][0]["passage"] == first.text

    invalid_id = validate_generation_result(
        {
            "decision": "answered",
            "answer": "30 days",
            "citations": [{"chunk_id": "not-retrieved", "quote": "30 days"}],
        },
        [first],
    )
    invalid_quote = validate_generation_result(
        {
            "decision": "answered",
            "answer": "30 days",
            "citations": [{"chunk_id": first.chunk_id, "quote": "90 days"}],
        },
        [first],
    )
    invalid_source = validate_generation_result(
        {
            "decision": "answered",
            "answer": "30 days",
            "selected_source": "not-retrieved.md",
            "citations": [{"chunk_id": first.chunk_id, "quote": "30 days"}],
        },
        [first],
    )
    assert invalid_id["decision"] == "insufficient_evidence"
    assert invalid_quote["decision"] == "insufficient_evidence"
    assert invalid_source["decision"] == "insufficient_evidence"
    assert invalid_id["citations"] == invalid_quote["citations"] == []


def test_format_result_retains_full_passages_and_same_file_chunks():
    candidates = [
        make_candidate(text="First complete passage, including all of its conditions."),
        make_candidate(
            chunk_id="doc-a:chunk:1",
            text="Second complete passage, including a separate exception.",
            chunk_index=1,
        ),
    ]
    response = app.format_result("Question", {
        "answer": "Answer",
        "source": "doc-a.md",
        "decision": "answered",
        "citations": [],
        "candidates": [candidate.__dict__ for candidate in candidates],
    })
    assert len(response.top_sources) == 2
    assert [item.passage for item in response.top_sources] == [item.text for item in candidates]
    assert response.top_sources[0].status_inferred is True
    assert response.top_sources[0].authority_score_inferred is True
    assert response.top_sources[0].scope_score_kind == "similarity"


def test_superseding_language_does_not_mark_replacement_superseded():
    assert detect_status("This policy supersedes all previous policy versions.") == "unknown"
    assert detect_status("This policy was superseded by the 2026 version.") == "superseded"


def test_same_document_conflict_is_not_suppressed_and_context_is_forwarded():
    first = Chunk(
        filename="one-policy.md",
        document_id="one-policy",
        chunk_index=0,
        chunk_id="one-policy:chunk:0",
        text="The maximum allowance is 30 days.",
        embedding=np.array([1.0, 0.0]),
        scope_embedding=np.array([1.0, 0.0]),
    )
    second = Chunk(
        filename="one-policy.md",
        document_id="one-policy",
        chunk_index=1,
        chunk_id="one-policy:chunk:1",
        text="The maximum allowance is 60 days.",
        embedding=np.array([1.0, 0.0]),
        scope_embedding=np.array([1.0, 0.0]),
    )

    class Embedder:
        def encode(self, texts, task_type=None):
            self.last_texts = texts
            return np.array([[1.0, 0.0]])

    class Generator:
        def generate(self, question, candidates, context=None, baseline=False):
            self.received_context = context
            return validate_generation_result({
                "decision": "unresolved_conflict",
                "answer": "The same document contains incompatible limits; its text does not resolve them.",
                "citations": [
                    {"chunk_id": candidate.chunk_id, "quote": candidate.text, "treatment": "conflicting"}
                    for candidate in candidates
                ],
                "explanation": "Both passages are in the same applicable source and no precedence is stated.",
            }, candidates)

    rag = GovRAGGemini.__new__(GovRAGGemini)
    rag.use_llm = True
    rag.embedder = Embedder()
    rag.generator = Generator()
    rag.chunks = [first, second]
    rag.doc_meta = {"one-policy.md": DocumentMeta("unknown", "unknown", 0.0)}

    result = rag.query("What is the maximum allowance?", context={"location": "North"})
    assert result["decision"] == "unresolved_conflict"
    assert result["conflict_detected"] is True
    assert len(result["candidates"]) == 2
    assert rag.generator.received_context == {"location": "North"}
    assert "location: North" in rag.embedder.last_texts[0]


def test_needs_information_outcome_preserves_missing_facts_and_question():
    candidate = make_candidate(text="Employees in Region North may claim 30 days.")
    result = validate_generation_result({
        "decision": "needs_information",
        "answer": "The rule depends on the employee's region.",
        "citations": [{"chunk_id": candidate.chunk_id, "quote": "in Region North", "treatment": "uncertain"}],
        "missing_information": ["Employee region"],
        "clarification_question": "Which region does the employee work in?",
        "explanation": "The location determines whether this rule applies.",
    }, [candidate])
    assert result["decision"] == "needs_information"
    assert result["missing_information"] == ["Employee region"]
    assert result["clarification_question"]


def test_same_number_exception_passages_are_not_collapsed():
    base = make_candidate(
        text="Employees may claim 30 days, except under seasonal employment terms.",
        claim_sentence="Employees may claim 30 days, except under seasonal employment terms.",
    )
    leave_exception = make_candidate(
        chunk_id="doc-a:chunk:1",
        claim_sentence="Employees may claim 30 days, except during approved leave.",
        chunk_index=1,
    )
    service_exception = make_candidate(
        chunk_id="doc-a:chunk:2",
        text="Employees may claim 30 days, except after service termination.",
        claim_sentence="Employees may claim 30 days, except after service termination.",
        chunk_index=2,
    )
    result = diversify_conflicts([base, leave_exception, service_exception], max_per_claim=1)
    assert {candidate.chunk_id for candidate in result} == {
        base.chunk_id, leave_exception.chunk_id, service_exception.chunk_id,
    }


def test_generation_failure_is_transparent_not_a_selected_winner():
    class BrokenModel:
        def generate_content(self, *args, **kwargs):
            raise RuntimeError("offline test failure")

    generator = GeminiGenerator.__new__(GeminiGenerator)
    generator.use_vertex = True
    generator.model = BrokenModel()
    result = generator.generate("question", [make_candidate()])
    assert result["decision"] == "insufficient_evidence"
    assert result["selected_source"] is None
    assert result["citations"] == []
    assert "Generation failed" in result["reason"]


def test_chunk_and_persisted_rows_keep_page_and_section_locations(tmp_path, monkeypatch):
    segments = chunk_text_with_locations("## Eligibility\nPage one text.\f## Exceptions\nPage two text.")
    assert [(segment.page_start, segment.section) for segment in segments] == [
        (1, "Eligibility"), (2, "Exceptions"),
    ]
    chunks = build_chunks([Document("policy__stable-id.md", "## Eligibility\nA rule.\f## Exceptions\nAn exception.")])
    rag = GovRAGGemini.__new__(GovRAGGemini)
    rag.chunks = chunks
    rag.doc_meta = {"policy__stable-id.md": DocumentMeta("unknown", "unknown", 0.0)}
    for chunk in rag.chunks:
        chunk.embedding = np.array([1.0, 0.0])
        chunk.scope_embedding = np.array([0.0, 1.0])
    rows = rag.to_persisted_index_rows()
    assert rows[0]["document_id"] == "stable-id"
    assert rows[0]["chunk_id"] == "stable-id:chunk:0"
    assert rows[0]["page_start"] == 1
    assert rows[1]["page_start"] == 2
    assert rows[1]["section"] == "Exceptions"
    store = CorpusStore(str(tmp_path))
    store.write_corpus_index("local-test", rows, {"index_format_version": 3})
    persisted_rows = store.load_corpus_index("local-test")["chunks"]
    assert persisted_rows[1]["page_start"] == 2
    assert persisted_rows[1]["section"] == "Exceptions"

    monkeypatch.setattr(rag_module, "GeminiEmbedder", lambda **kwargs: object())
    restored = GovRAGGemini.from_persisted_index(persisted_rows, use_llm=False)
    assert restored.chunks[1].page_start == 2
    assert restored.chunks[1].section == "Exceptions"


def test_old_persisted_indexes_are_rejected_for_rebuild():
    current = {
        "corpus_updated_at": "now",
        "embedding_model": app.VERTEX_AI_MODEL,
        "index_format_version": app.PERSISTED_INDEX_FORMAT_VERSION,
    }
    valid_rows = [{
        "chunk_id": "doc:chunk:0",
        "document_id": "doc",
        "scope_embedding": [1.0],
        "page_start": None,
        "page_end": None,
        "section": None,
    }]
    assert app.persisted_index_is_current(current, "now", valid_rows)
    current["index_format_version"] = 2
    assert not app.persisted_index_is_current(current, "now", valid_rows)
    assert not app.persisted_index_is_current(
        {**current, "index_format_version": app.PERSISTED_INDEX_FORMAT_VERSION},
        "now",
        [{
            "document_id": "old-doc",
            "scope_embedding": [1.0],
            "page_start": None,
            "page_end": None,
            "section": None,
        }],
    )


def test_ellipsis_joined_quotes_verify_and_bad_citations_are_named():
    cand = make_candidate(
        text="Scientists say humans will never live beyond 150 years. Other work disagrees. "
             "The longest any human is likely to live is 150 years.",
        claim_sentence="The longest any human is likely to live is 150 years.",
    )
    ok = validate_generation_result(
        {
            "decision": "answered",
            "answer": "150 years",
            "citations": [{"chunk_id": cand.chunk_id,
                           "quote": "humans will never live beyond 150 years... the longest any human is likely to live is 150 years",
                           "treatment": "governing"}],
        },
        [cand],
    )
    assert ok["decision"] == "answered" and len(ok["citations"]) == 1 and ok["dropped_citations"] == 0

    out_of_order = validate_generation_result(
        {
            "decision": "answered",
            "answer": "150 years",
            "citations": [{"chunk_id": cand.chunk_id,
                           "quote": "likely to live is 150 years ... humans will never live", "treatment": "governing"}],
        },
        [cand],
    )
    assert out_of_order["decision"] == "insufficient_evidence"
    assert "citation 1" in out_of_order["reason"] and cand.chunk_id in out_of_order["reason"]

    salvaged = validate_generation_result(
        {
            "decision": "answered",
            "answer": "150 years",
            "selected_source": "ghost.md",
            "citations": [
                {"chunk_id": cand.chunk_id, "quote": "never live beyond 150 years", "treatment": "governing"},
                {"chunk_id": "ghost:chunk:0", "quote": "anything", "treatment": "supporting"},
            ],
        },
        [cand],
        drop_unverifiable=True,
    )
    assert salvaged["decision"] == "answered"
    assert len(salvaged["citations"]) == 1 and salvaged["dropped_citations"] == 1
    assert salvaged["selected_source"] == cand.filename
