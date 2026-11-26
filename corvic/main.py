"""GOV-RAG FastAPI service with dynamic multi-corpus support."""

from __future__ import annotations

import glob
import io
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from pipeline.corpus_store import (
        CorpusNotFoundError,
        CorpusStore,
        CorpusStoreError,
        DocumentNotFoundError,
    )
    from pipeline.document_parser import (
        SUPPORTED_EXTENSIONS,
        DocumentParseError,
        make_normalized_filename,
        parse_document_bytes,
        sanitize_filename,
        validate_extension,
    )
    from pipeline.gov_rag_gemini import (
        DocumentMeta,
        GovRAGGemini,
        VERTEX_AI_MODEL,
        authority_score,
        detect_doc_type,
        detect_status,
        normalize_claim,
    )
    from pipeline.evidence_locator import locate_chunks, locate_quote
    from pipeline.providers import LocalEmbedder, OpenAICompatGenerator
    from pipeline.rate_limit import RateLimiter
except ModuleNotFoundError as exc:
    if exc.name != "pipeline":
        raise
    from .pipeline.corpus_store import (
        CorpusNotFoundError,
        CorpusStore,
        CorpusStoreError,
        DocumentNotFoundError,
    )
    from .pipeline.document_parser import (
        SUPPORTED_EXTENSIONS,
        DocumentParseError,
        make_normalized_filename,
        parse_document_bytes,
        sanitize_filename,
        validate_extension,
    )
    from .pipeline.gov_rag_gemini import (
        DocumentMeta,
        GovRAGGemini,
        VERTEX_AI_MODEL,
        authority_score,
        detect_doc_type,
        detect_status,
        normalize_claim,
    )
    from .pipeline.evidence_locator import locate_chunks, locate_quote
    from .pipeline.providers import LocalEmbedder, OpenAICompatGenerator
    from .pipeline.rate_limit import RateLimiter


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env_file(path: str) -> None:
    """Minimal KEY=VALUE loader for corvic/.env (never prints values)."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(os.path.join(BASE_DIR, ".env"))

API_SECRET = os.getenv("GOVRAG_API_SECRET")
MAX_FILE_SIZE_MB = int(os.getenv("GOVRAG_MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Server-side folder connector: a folder path typed in the UI is indexed in place.
# It must sit under one of these roots (comma-separated). Default: the home
# directory of the account running the server. Set to "" to disable the connector.
_ROOTS_ENV = os.getenv("DOCUINSIGHT_ALLOWED_ROOTS", "~")
ALLOWED_SOURCE_ROOTS = [
    os.path.realpath(os.path.expanduser(root.strip()))
    for root in _ROOTS_ENV.split(",") if root.strip()
]

DEMO_DOCUMENTS_DIR = os.path.join(BASE_DIR, "easy")
STATIC_DIR = os.path.join(BASE_DIR, "static")
GOVRAG_BUCKET = os.getenv("GOVRAG_BUCKET", "").strip() or None
PERSISTED_INDEX_FORMAT_VERSION = 3

# Server-controlled Vertex AI target for the public browser API; never client-supplied.
DEMO_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "project-79920195-9e86-44ea-8c9")
DEMO_REGION = os.getenv("VERTEX_AI_REGION", "us-central1")

# ---- Pipeline configuration (the paper's "Variant 3" is the default) --------
# GOVRAG_PROVIDER=nvidia: local bge embeddings + any OpenAI-compatible chat
# endpoint (NVIDIA hosted Nemotron by default). GOVRAG_PROVIDER=gemini keeps the
# original Vertex AI path.
GOVRAG_PROVIDER = os.getenv("GOVRAG_PROVIDER", "nvidia" if (os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")) else "gemini").lower()
GOVRAG_RETRIEVAL = os.getenv("GOVRAG_RETRIEVAL", "hybrid")           # hybrid | dense
GOVRAG_POLICY = os.getenv("GOVRAG_POLICY", "answer_and_flag")        # answer_and_flag | strict
GENERATION_MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
EMBED_MODEL = os.getenv("GOVRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GOVRAG_BASE_URL = os.getenv("DOCUINSIGHT_BASE_URL") or os.getenv("GOVRAG_BASE_URL") or None
assert GOVRAG_RETRIEVAL in ("hybrid", "dense"), GOVRAG_RETRIEVAL
assert GOVRAG_POLICY in ("answer_and_flag", "strict"), GOVRAG_POLICY

STATUS_OPTIONS = ["active_authoritative", "active", "draft", "superseded", "archived", "unknown"]
DOC_TYPE_OPTIONS = [
    "policy", "contract", "guidance", "memo", "report", "faq", "technical_document",
    "documentation", "research_paper", "secondary_summary", "unknown",
]
# Authority adjustments applied when a user overrides a document's status.
STATUS_AUTHORITY_FLOOR = {"active_authoritative": 10.0, "active": 6.0}
STATUS_AUTHORITY_CEILING = {"draft": -3.0, "superseded": -6.0, "archived": -6.0}

DEMO_CORPUS_TTL_HOURS = float(os.getenv("DEMO_CORPUS_TTL_HOURS", "24"))
MAX_DEMO_DOCUMENTS = int(os.getenv("GOVRAG_MAX_DEMO_DOCUMENTS", "500"))
MAX_DEMO_CORPUS_MB = int(os.getenv("GOVRAG_MAX_DEMO_CORPUS_MB", "500"))
MAX_DEMO_CORPUS_BYTES = MAX_DEMO_CORPUS_MB * 1024 * 1024
MAX_QUESTION_LENGTH = int(os.getenv("GOVRAG_MAX_QUESTION_LENGTH", "2000"))

DEMO_RATE_LIMIT_PER_MINUTE = int(os.getenv("GOVRAG_DEMO_RATE_LIMIT_PER_MINUTE", "30"))
demo_rate_limiter = RateLimiter(max_requests=DEMO_RATE_LIMIT_PER_MINUTE, window_seconds=60.0)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("GOVRAG_ALLOWED_ORIGINS", "https://app.corvic.ai").split(",")
    if origin.strip()
]
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="DocResolve API",
    description="Authority-aware RAG with conflict resolution",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class CaseContext(BaseModel):
    relevant_date: Optional[str] = None
    location: Optional[str] = None
    employee_category: Optional[str] = None
    circumstances: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    project_id: str = "project-79920195-9e86-44ea-8c9"
    region: str = "us-central1"
    use_llm: bool = True
    top_k: int = 8
    context: Optional[CaseContext] = None


class CorpusCreateRequest(BaseModel):
    name: str


class CorpusIndexRequest(BaseModel):
    project_id: str = "project-79920195-9e86-44ea-8c9"
    region: str = "us-central1"
    use_llm: bool = True


class DemoCorpusCreateRequest(BaseModel):
    name: str = "Demo Workspace"


class DemoQueryRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LENGTH)
    context: Optional[CaseContext] = None
    top_k: int = Field(8, ge=1, le=12)


class DocumentOverrideRequest(BaseModel):
    status_override: Optional[str] = None
    doc_type_override: Optional[str] = None
    clear: bool = False


class FolderSourceRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=4096)
    label: Optional[str] = Field(default=None, max_length=200)


class SourceInfo(BaseModel):
    filename: str
    original_filename: str
    document_id: str
    chunk_id: str
    passage: str
    claim: Optional[str] = None
    status: str
    status_inferred: bool = True
    document_type: str
    document_type_inferred: bool = True
    semantic_score: float
    authority_score: float
    authority_score_inferred: bool = True
    scope_score: float
    scope_score_kind: str = "similarity"
    final_score: float
    supporting_sentence: Optional[str] = None
    selected: bool = False
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None


class CitationInfo(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    passage: str
    quote: str
    treatment: str
    explanation: str = ""
    # Character spans of the verified quote (one per ellipsis segment).
    passage_spans: List[List[int]] = Field(default_factory=list)
    document_spans: List[List[int]] = Field(default_factory=list)
    role: str = "support"  # support | dissent | context


class CorroborationEntry(BaseModel):
    value: str
    documents: List[str]
    document_ids: List[str]
    count: int
    committed: bool


class ResolutionInfo(BaseModel):
    rule: str  # governance | corroboration | consistency | tie | single_source | none
    summary: str
    committed_values: List[str] = Field(default_factory=list)
    corroboration: List[CorroborationEntry] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    selected_source: Optional[str]
    reason: str
    conflict_detected: bool
    confidence: str
    num_conflicts: int
    top_sources: List[SourceInfo]
    retrieval_mode: Optional[str] = None
    decision: str = "insufficient_evidence"
    explanation: str = ""
    citations: List[CitationInfo] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    short_answers: List[str] = Field(default_factory=list)
    decision_label: str = "insufficient_evidence"  # adds answered_with_dissent
    resolution: Optional[ResolutionInfo] = None
    latency_s: Optional[float] = None
    model: Optional[str] = None
    pipeline: Dict[str, str] = Field(default_factory=dict)


class CorpusQueryResponse(QueryResponse):
    corpus_id: str


# ============================================================
# GLOBAL STATE
# ============================================================

demo_rag_instance = None
demo_rag_config = None

corpus_store = CorpusStore(base_dir=BASE_DIR, bucket_name=GOVRAG_BUCKET)

# Cache entries keyed by "corpus_id|project|region|use_llm".
rag_instances: Dict[str, GovRAGGemini] = {}
rag_instance_metadata: Dict[str, Dict] = {}

cache_lock = threading.Lock()
corpus_locks: Dict[str, threading.Lock] = {}

# Document text cache for the reader pane: (corpus_id, document_id) -> text.
document_text_cache: Dict[str, str] = {}


class _LockedEmbedder:
    """Serialize encode() calls; the sentence-transformers model is not thread-safe."""

    def __init__(self, inner):
        self.inner = inner
        self.model_name = getattr(inner, "model_name", EMBED_MODEL)
        self._lock = threading.Lock()

    def encode(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        with self._lock:
            return self.inner.encode(texts, task_type=task_type)


_shared_embedder = None
_shared_generator = None
_provider_lock = threading.Lock()


def get_providers():
    """Return (embedder, generator) for the configured provider, or (None, None) for Gemini."""
    global _shared_embedder, _shared_generator
    if GOVRAG_PROVIDER != "nvidia":
        return None, None
    with _provider_lock:
        if _shared_embedder is None:
            _shared_embedder = _LockedEmbedder(LocalEmbedder(model_name=EMBED_MODEL))
        if _shared_generator is None:
            generator_kwargs = {"base_url": GOVRAG_BASE_URL} if GOVRAG_BASE_URL else {}
            _shared_generator = OpenAICompatGenerator(model=GENERATION_MODEL, **generator_kwargs)
    return _shared_embedder, _shared_generator


def embedding_model_name() -> str:
    return EMBED_MODEL if GOVRAG_PROVIDER == "nvidia" else VERTEX_AI_MODEL


def pipeline_description() -> Dict[str, str]:
    return {
        "provider": GOVRAG_PROVIDER,
        "generation_model": GENERATION_MODEL if GOVRAG_PROVIDER == "nvidia" else "gemini-2.5-pro",
        "embedding_model": embedding_model_name(),
        "retrieval": GOVRAG_RETRIEVAL,
        "policy": GOVRAG_POLICY,
    }


# ============================================================
# HELPERS
# ============================================================

def require_api_key(x_api_key: Optional[str]):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def validate_corpus_id(corpus_id: str) -> None:
    try:
        parsed = uuid.UUID(corpus_id)
        if str(parsed) != corpus_id:
            raise ValueError("mismatch")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid corpus_id") from exc


def get_corpus_lock(corpus_id: str) -> threading.Lock:
    with cache_lock:
        if corpus_id not in corpus_locks:
            corpus_locks[corpus_id] = threading.Lock()
        return corpus_locks[corpus_id]


def build_cache_key(corpus_id: str, request: QueryRequest) -> str:
    return f"{corpus_id}|{request.project_id}|{request.region}|{int(request.use_llm)}"


def persisted_index_is_current(index_metadata: Dict, corpus_updated_at: Optional[str], rows: List[Dict]) -> bool:
    return (
        index_metadata.get("corpus_updated_at") == corpus_updated_at
        and index_metadata.get("embedding_model") == embedding_model_name()
        and index_metadata.get("index_format_version") == PERSISTED_INDEX_FORMAT_VERSION
        and bool(rows)
        and all(row.get("scope_embedding") is not None for row in rows)
        and all(row.get("chunk_id") and row.get("document_id") for row in rows)
        and all("page_start" in row and "page_end" in row and "section" in row for row in rows)
    )


def invalidate_corpus_cache(corpus_id: str) -> None:
    with cache_lock:
        keys = [k for k in rag_instances if k.startswith(f"{corpus_id}|")]
        for key in keys:
            rag_instances.pop(key, None)
            rag_instance_metadata.pop(key, None)


def get_client_key(http_request: Request) -> str:
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def enforce_demo_rate_limit(http_request: Request) -> None:
    if not demo_rate_limiter.allow(get_client_key(http_request)):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")


def get_demo_corpus_or_404(corpus_id: str) -> Dict:
    """Resolve a demo-kind corpus, treating expired/foreign-kind corpora as not found."""
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.get_corpus(corpus_id)
    except CorpusNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Corpus not found") from exc

    if metadata.get("kind") != "demo":
        raise HTTPException(status_code=404, detail="Corpus not found")

    if corpus_store.is_expired(metadata):
        try:
            corpus_store.delete_corpus(corpus_id)
        except CorpusStoreError:
            pass
        invalidate_corpus_cache(corpus_id)
        raise HTTPException(status_code=404, detail="Corpus not found or expired")

    return metadata


def get_demo_rag_instance(request: QueryRequest) -> GovRAGGemini:
    global demo_rag_instance, demo_rag_config

    if not os.path.exists(DEMO_DOCUMENTS_DIR):
        raise HTTPException(
            status_code=500,
            detail=f"Documents directory not found: {DEMO_DOCUMENTS_DIR}",
        )

    config = (request.project_id, request.region, request.use_llm)
    if demo_rag_instance is None or demo_rag_config != config:
        print("Initializing demo GOV-RAG...")
        print(f"Documents: {DEMO_DOCUMENTS_DIR}")

        embedder, generator = get_providers()
        demo_rag_instance = GovRAGGemini(
            doc_dir=DEMO_DOCUMENTS_DIR,
            project_id=request.project_id,
            region=request.region,
            use_llm=request.use_llm,
            embedder=embedder,
            generator=generator,
        )
        demo_rag_config = config
        print("Demo GOV-RAG initialized.")

    return demo_rag_instance


def ensure_corpus_rag(corpus_id: str, request: QueryRequest, force_reindex: bool = False):
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.get_corpus(corpus_id)
    except CorpusNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Corpus not found") from exc

    if metadata.get("document_count", 0) == 0:
        raise HTTPException(status_code=409, detail="Corpus has no documents")

    key = build_cache_key(corpus_id, request)
    lock = get_corpus_lock(corpus_id)

    with lock:
        metadata = corpus_store.get_corpus(corpus_id)
        cached_meta = rag_instance_metadata.get(key)
        can_reuse = (
            not force_reindex
            and key in rag_instances
            and cached_meta is not None
            and cached_meta.get("updated_at") == metadata.get("updated_at")
        )

        if can_reuse:
            print(f"govrag_index_source=memory_cache corpus_id={corpus_id}")
            return rag_instances[key], cached_meta

        if not force_reindex:
            try:
                persisted = corpus_store.load_corpus_index(corpus_id)
                index_meta = persisted["metadata"]
                corpus_updated_at = metadata.get("updated_at")
                index_updated_at = index_meta.get("corpus_updated_at")
                index_model = index_meta.get("embedding_model")
                index_version = index_meta.get("index_format_version")
                rows = persisted.get("chunks", [])
                has_scope_embeddings = all(row.get("scope_embedding") is not None for row in rows)

                if persisted_index_is_current(index_meta, corpus_updated_at, rows):
                    embedder, generator = get_providers()
                    rag = GovRAGGemini.from_persisted_index(
                        persisted["chunks"],
                        project_id=request.project_id,
                        region=request.region,
                        use_llm=request.use_llm,
                        embedder=embedder,
                        generator=generator,
                    )
                    apply_governance_overrides(rag, metadata)
                    cache_meta = {
                        "filename_map": index_meta.get("filename_map", {}),
                        "local_dir": None,
                        "updated_at": corpus_updated_at,
                        "document_count": metadata.get("document_count", 0),
                    }
                    rag_instances[key] = rag
                    rag_instance_metadata[key] = cache_meta
                    print(f"govrag_index_source=persisted_index corpus_id={corpus_id}")
                    return rag, cache_meta

                if index_updated_at != corpus_updated_at:
                    print(
                        "govrag_persisted_index_invalid=updated_at_changed "
                        f"corpus_id={corpus_id} "
                        f"index_updated_at={index_updated_at} "
                        f"corpus_updated_at={corpus_updated_at}"
                    )
                elif index_model != embedding_model_name():
                    print(
                        "govrag_persisted_index_invalid=embedding_model_changed "
                        f"corpus_id={corpus_id} "
                        f"index_model={index_model} "
                        f"current_model={embedding_model_name()}"
                    )
                elif index_version != PERSISTED_INDEX_FORMAT_VERSION:
                    print(
                        "govrag_persisted_index_invalid=index_format_changed "
                        f"corpus_id={corpus_id} "
                        f"index_format_version={index_version} "
                        f"current_format_version={PERSISTED_INDEX_FORMAT_VERSION}"
                    )
                elif not has_scope_embeddings:
                    print(
                        "govrag_persisted_index_invalid=missing_scope_embeddings "
                        f"corpus_id={corpus_id}"
                    )
                elif not all(row.get("chunk_id") and row.get("document_id") for row in rows):
                    print(
                        "govrag_persisted_index_invalid=missing_chunk_identity "
                        f"corpus_id={corpus_id}"
                    )
                else:
                    print(
                        "govrag_persisted_index_invalid=missing_locations "
                        f"corpus_id={corpus_id}"
                    )
            except CorpusStoreError:
                pass

        materialized = corpus_store.materialize_corpus(corpus_id)
        embedder, generator = get_providers()
        rag = GovRAGGemini(
            doc_dir=materialized["local_dir"],
            project_id=request.project_id,
            region=request.region,
            use_llm=request.use_llm,
            embedder=embedder,
            generator=generator,
        )
        apply_governance_overrides(rag, metadata)

        ready_meta = corpus_store.mark_ready(corpus_id)
        corpus_store.write_corpus_index(
            corpus_id,
            rag.to_persisted_index_rows(),
            {
                "index_format_version": PERSISTED_INDEX_FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "corpus_id": corpus_id,
                "corpus_updated_at": ready_meta.get("updated_at"),
                "document_count": ready_meta.get("document_count", 0),
                "embedding_model": embedding_model_name(),
                "filename_map": materialized["filename_map"],
            },
        )
        cache_meta = {
            "filename_map": materialized["filename_map"],
            "local_dir": materialized["local_dir"],
            "updated_at": ready_meta.get("updated_at"),
            "document_count": ready_meta.get("document_count", 0),
        }

        rag_instances[key] = rag
        rag_instance_metadata[key] = cache_meta
        print(f"govrag_index_source=rebuilt_index corpus_id={corpus_id}")
        return rag, cache_meta


def apply_governance_overrides(rag: GovRAGGemini, metadata: Dict) -> None:
    """Replace inferred status/type with the user's overrides for this corpus.

    doc_meta is keyed by the normalized filename the pipeline loaded. Authority is
    nudged so an override actually changes ranking, not only the label shown to
    the model.
    """
    for doc in metadata.get("documents", []):
        status_override = doc.get("status_override")
        type_override = doc.get("doc_type_override")
        if not status_override and not type_override:
            continue
        key = doc.get("normalized_filename")
        current = rag.doc_meta.get(key) or DocumentMeta(status="unknown", doc_type="unknown", authority_raw=0.0)
        status = status_override or current.status
        authority = current.authority_raw
        if status_override in STATUS_AUTHORITY_FLOOR:
            authority = max(authority, STATUS_AUTHORITY_FLOOR[status_override])
        if status_override in STATUS_AUTHORITY_CEILING:
            authority = min(authority, STATUS_AUTHORITY_CEILING[status_override])
        # The evidence block shows "Inferred status: ..."; mark reviewer decisions
        # so the model weighs them above cues it reads in the text itself.
        rag.doc_meta[key] = DocumentMeta(
            status=f"{status} (set by reviewer)" if status_override else status,
            doc_type=f"{type_override} (set by reviewer)" if type_override else current.doc_type,
            authority_raw=authority,
        )


def get_document_text(corpus_id: Optional[str], document_id: str, rag: Optional[GovRAGGemini] = None) -> Optional[str]:
    """Full normalized text of a document, from the store (cached) or from the loaded chunks."""
    if corpus_id:
        key = f"{corpus_id}|{document_id}"
        text = document_text_cache.get(key)
        if text is None:
            try:
                text = corpus_store.read_document_text(corpus_id, document_id)
            except CorpusStoreError:
                text = None
            if text is not None:
                document_text_cache[key] = text
        if text is not None:
            return text
    if rag is not None:
        for doc in getattr(rag, "documents", []) or []:
            if doc.filename == document_id or getattr(doc, "document_id", None) == document_id:
                return doc.text
    return None


def _norm_value(value: str) -> str:
    return normalize_claim(value) if value else ""


def build_resolution(result: dict, candidates: List[dict], citations: List[dict], filename_map: Dict[str, str]) -> ResolutionInfo:
    """Summarize how the disagreement was (or was not) resolved.

    Corroboration counts distinct documents per numeric claim value among the
    evidence the model saw. The rule is inferred from decision, treatments and
    governance signals; it is a reading aid, not a model output.
    """
    decision = result.get("decision", "insufficient_evidence")
    committed = [str(v) for v in (result.get("short_answers") or []) if str(v).strip()]
    committed_norm = {_norm_value(v) for v in committed}

    by_claim: Dict[str, Dict] = {}
    for cand in candidates:
        claim = cand.get("claim")
        key = _norm_value(str(claim)) if claim is not None else "NO_CLAIM"
        if key == "NO_CLAIM" or not key:
            continue
        if re.fullmatch(r"(19|20)\d\d", key):
            continue  # bare years are dates, not competing values
        entry = by_claim.setdefault(key, {"value": str(claim), "documents": [], "document_ids": []})
        name = filename_map.get(cand.get("filename", ""), cand.get("filename", ""))
        doc_id = str(cand.get("document_id") or cand.get("filename") or "")
        if doc_id not in entry["document_ids"]:
            entry["documents"].append(name)
            entry["document_ids"].append(doc_id)

    def is_committed(key: str, value: str) -> bool:
        if key in committed_norm:
            return True
        return any(key and (key in c or c in key) for c in committed_norm if c)

    # Only values the model engaged with: committed ones, or ones found in a
    # cited document. Other numeric claims among the retrieved chunks are noise
    # from unrelated passages in large collections.
    cited_doc_ids = {str(c.get("document_id") or c.get("filename") or "") for c in citations}
    corroboration = sorted(
        (
            CorroborationEntry(
                value=e["value"], documents=e["documents"], document_ids=e["document_ids"],
                count=len(e["document_ids"]), committed=is_committed(k, e["value"]),
            )
            for k, e in by_claim.items()
            if is_committed(k, e["value"]) or (set(e["document_ids"]) & cited_doc_ids)
        ),
        key=lambda e: (not e.committed, -e.count, e.value),
    )

    treatments = {c.get("treatment") for c in citations}
    governing = [c for c in citations if c.get("treatment") == "governing"]
    dissent = [c for c in citations if c.get("treatment") in ("conflicting", "superseded", "different_scope")]
    cited_committed = [e for e in corroboration if e.committed]
    cited_other = [e for e in corroboration if not e.committed]

    if decision == "unresolved_conflict":
        rule, summary = "tie", "Competing values are equally supported; the system did not choose."
    elif decision in ("insufficient_evidence", "needs_information"):
        rule, summary = "none", "No value was committed."
    elif not result.get("conflict_detected") and not dissent:
        rule = "single_source" if len({c.get("document_id") for c in citations}) <= 1 else "consistency"
        summary = "The cited evidence agrees." if rule == "consistency" else "Answered from one source; no disagreement found."
    elif "superseded" in treatments or any(
        (c.get("status") or "").startswith("active") for c in candidates if c.get("filename") in {g.get("filename") for g in governing}
    ):
        rule, summary = "governance", "An active or authoritative document outranks the dissenting one."
    elif cited_committed and cited_other and max(e.count for e in cited_committed) > max(e.count for e in cited_other):
        top, other = max(e.count for e in cited_committed), max(e.count for e in cited_other)
        rule, summary = "corroboration", f"The committed value is backed by {top} independent document{'s' if top != 1 else ''} against {other}."
    else:
        rule, summary = "consistency", "The committed value is the one consistent with the surrounding evidence; the dissent is flagged."

    return ResolutionInfo(rule=rule, summary=summary, committed_values=committed, corroboration=corroboration)


def format_result(question: str, result: dict, top_k: int = 8, filename_map: Optional[Dict[str, str]] = None,
                  corpus_id: Optional[str] = None, rag: Optional[GovRAGGemini] = None, latency_s: Optional[float] = None):
    """Map chunk-level evidence and verified citations to the public response."""
    filename_map = filename_map or {}
    selected_source_raw = result.get("source")
    selected_source = filename_map.get(selected_source_raw, selected_source_raw)
    conflict_detected = result.get("conflict_detected", False)
    candidates = result.get("candidates", [])
    ordered_candidates = sorted(
        candidates, key=lambda candidate: candidate.get("final_score", 0.0), reverse=True
    )

    top_sources = []
    for src in ordered_candidates[:top_k]:
        raw_filename = src.get("filename", "")
        display_filename = filename_map.get(raw_filename, raw_filename)
        chunk_index = src.get("chunk_index", 0)
        top_sources.append(
            SourceInfo(
                filename=display_filename,
                original_filename=display_filename,
                document_id=str(src.get("document_id") or raw_filename),
                chunk_id=str(src.get("chunk_id") or f"{raw_filename}:chunk:{chunk_index}"),
                passage=str(src.get("text") or ""),
                claim=src.get("claim"),
                status=src.get("status", "unknown"),
                document_type=src.get("doc_type", "unknown"),
                semantic_score=float(src.get("semantic_score", 0.0)),
                authority_score=float(src.get("authority_score", 0.0)),
                scope_score=float(src.get("scope_score", 0.0)),
                final_score=float(src.get("final_score", 0.0)),
                supporting_sentence=src.get("claim_sentence"),
                selected=bool(selected_source_raw and raw_filename == selected_source_raw),
                page_start=src.get("page_start"),
                page_end=src.get("page_end"),
                section=src.get("section"),
            )
        )

    citations = []
    for citation in result.get("citations", []):
        raw_filename = citation.get("filename", "")
        display_filename = filename_map.get(raw_filename, raw_filename)
        document_id = str(citation.get("document_id") or raw_filename)
        passage = citation.get("passage", "") or ""
        quote = citation["quote"]
        treatment = citation.get("treatment", "supporting")
        passage_spans = [list(span) for span in locate_quote(quote, passage)]
        document_spans: List[List[int]] = []
        document_text = get_document_text(corpus_id, document_id, rag)
        if document_text:
            document_spans = [list(span) for span in locate_quote(quote, document_text)]
        role = "dissent" if treatment in ("conflicting", "superseded", "different_scope") else (
            "context" if treatment == "uncertain" else "support")
        citations.append(CitationInfo(
            chunk_id=citation["chunk_id"],
            document_id=document_id,
            filename=display_filename,
            page_start=citation.get("page_start"),
            page_end=citation.get("page_end"),
            section=citation.get("section"),
            passage=passage,
            quote=quote,
            treatment=treatment,
            explanation=citation.get("explanation", ""),
            passage_spans=passage_spans,
            document_spans=document_spans,
            role=role,
        ))

    if conflict_detected:
        claims = {
            str(src.get("claim")).strip()
            for src in ordered_candidates
            if src.get("claim") is not None
        }
        num_conflicts = max(len(claims) - 1, 0)
    else:
        num_conflicts = 0

    decision = result.get("decision", "insufficient_evidence")
    has_dissent = any(c.role == "dissent" for c in citations)
    decision_label = "answered_with_dissent" if decision == "answered" and (conflict_detected or has_dissent) else decision
    resolution = build_resolution(result, ordered_candidates, result.get("citations", []), filename_map)

    return QueryResponse(
        question=question,
        answer=result.get("answer"),
        selected_source=selected_source,
        reason=result.get("reason", result.get("explanation", "")) or "",
        conflict_detected=conflict_detected,
        confidence=result.get("confidence", "unknown"),
        num_conflicts=num_conflicts,
        top_sources=top_sources,
        retrieval_mode=result.get("retrieval_mode"),
        decision=decision,
        explanation=result.get("explanation", result.get("reason", "")) or "",
        citations=citations,
        missing_information=result.get("missing_information", []),
        clarification_question=result.get("clarification_question"),
        short_answers=[str(v) for v in (result.get("short_answers") or [])],
        decision_label=decision_label,
        resolution=resolution,
        latency_s=round(latency_s, 2) if latency_s is not None else None,
        model=pipeline_description()["generation_model"],
        pipeline=pipeline_description(),
    )


def run_demo_query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    rag = get_demo_rag_instance(request)
    context = request.context.model_dump(exclude_none=True) if request.context else None
    started = time.perf_counter()
    result = rag.query(question, context=context, retrieval=GOVRAG_RETRIEVAL, policy=GOVRAG_POLICY)
    return format_result(question, result, request.top_k, rag=rag, latency_s=time.perf_counter() - started)


def run_corpus_query(corpus_id: str, request: QueryRequest) -> CorpusQueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    rag, cache_meta = ensure_corpus_rag(corpus_id, request, force_reindex=False)
    context = request.context.model_dump(exclude_none=True) if request.context else None
    started = time.perf_counter()
    result = rag.query(question, context=context, retrieval=GOVRAG_RETRIEVAL, policy=GOVRAG_POLICY)
    formatted = format_result(
        question, result, request.top_k, filename_map=cache_meta.get("filename_map", {}),
        corpus_id=corpus_id, rag=rag, latency_s=time.perf_counter() - started,
    )

    return CorpusQueryResponse(corpus_id=corpus_id, **formatted.model_dump())


def map_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CorpusNotFoundError):
        return HTTPException(status_code=404, detail="Corpus not found")
    if isinstance(exc, DocumentNotFoundError):
        return HTTPException(status_code=404, detail="Document not found")
    if isinstance(exc, DocumentParseError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, CorpusStoreError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


# ============================================================
# PUBLIC ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "GOV-RAG API",
        "status": "running",
        "version": "2.0.0",
    }


# Static assets (app.css/app.js) for the DocuResolve frontend, same-origin as the API.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_for_frontend(request: Request, call_next):
    """The demo UI changes often; make browsers revalidate its HTML, script and styles."""
    response = await call_next(request)
    if request.url.path == "/app" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/app")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "app.html"))


@app.get("/health")
async def health():
    demo_docs_exist = os.path.exists(DEMO_DOCUMENTS_DIR)
    return {
        "status": "healthy" if demo_docs_exist else "degraded",
        "demo_documents_directory": DEMO_DOCUMENTS_DIR,
        "demo_documents_available": demo_docs_exist,
        "demo_rag_initialized": demo_rag_instance is not None,
        "corpus_backend": "gcs" if GOVRAG_BUCKET else "local",
        "govrag_bucket": GOVRAG_BUCKET,
        "pipeline": pipeline_description(),
        "status_options": STATUS_OPTIONS,
        "doc_type_options": DOC_TYPE_OPTIONS,
        "folder_connector": {"enabled": bool(ALLOWED_SOURCE_ROOTS), "roots": ALLOWED_SOURCE_ROOTS},
    }


@app.post("/demo_query", response_model=QueryResponse)
async def demo_query(request: QueryRequest):
    try:
        return run_demo_query(request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"DEMO QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# PROTECTED DEMO ENDPOINTS
# ============================================================

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    try:
        return run_demo_query(request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/batch_query")
async def batch_query(
    questions: List[str],
    project_id: str = "project-79920195-9e86-44ea-8c9",
    region: str = "us-central1",
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)

    try:
        request = QueryRequest(question="placeholder", project_id=project_id, region=region, use_llm=True, top_k=8)
        rag = get_demo_rag_instance(request)

        results = []
        for question in questions:
            clean_question = question.strip()
            if not clean_question:
                continue
            result = rag.query(clean_question, retrieval=GOVRAG_RETRIEVAL, policy=GOVRAG_POLICY)
            results.append(format_result(clean_question, result, 8, rag=rag))

        return results
    except HTTPException:
        raise
    except Exception as exc:
        print(f"BATCH ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# CORPUS ENDPOINTS (PROTECTED)
# ============================================================

@app.post("/corpora")
async def create_corpus(payload: CorpusCreateRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    try:
        corpus_id = str(uuid.uuid4())
        metadata = corpus_store.create_corpus(corpus_id=corpus_id, name=payload.name)
        return {
            "corpus_id": metadata["corpus_id"],
            "name": metadata["name"],
            "status": metadata["status"],
            "document_count": metadata["document_count"],
            "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/corpora")
async def list_corpora(x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    try:
        corpora = corpus_store.list_corpora()
        return {
            "count": len(corpora),
            "corpora": [
                {
                    "corpus_id": c.get("corpus_id"),
                    "name": c.get("name"),
                    "status": c.get("status"),
                    "document_count": c.get("document_count", 0),
                    "created_at": c.get("created_at"),
                    "updated_at": c.get("updated_at"),
                }
                for c in corpora
            ],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/corpora/{corpus_id}")
async def get_corpus(corpus_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.get_corpus(corpus_id)
        return {
            "corpus_id": metadata.get("corpus_id"),
            "name": metadata.get("name"),
            "status": metadata.get("status"),
            "document_count": metadata.get("document_count", 0),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "documents": metadata.get("documents", []),
        }
    except Exception as exc:
        raise map_store_error(exc)


CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def sanitize_relative_path(path: Optional[str], fallback: str) -> str:
    """Keep a folder upload's relative path, safe for display and storage keys."""
    if not path:
        return fallback
    parts = [p for p in path.replace("\\", "/").split("/") if p and p not in (".", "..")]
    parts = [re_sub_unsafe(p) for p in parts]
    parts = [p for p in parts if p]
    return "/".join(parts) if parts else fallback


def re_sub_unsafe(part: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._ -]", "_", part).strip()


def infer_governance(text: str) -> Dict[str, object]:
    return {
        "inferred_status": detect_status(text),
        "inferred_doc_type": detect_doc_type(text),
        "inferred_authority_raw": float(authority_score(text)),
    }


async def handle_document_uploads(
    corpus_id: str,
    files: List[UploadFile],
    max_files: Optional[int] = None,
    max_total_size_bytes: Optional[int] = None,
    relative_paths: Optional[List[str]] = None,
) -> Dict:
    """Shared upload path for both the protected /corpora and public /demo/corpora APIs."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    relative_paths = relative_paths or []

    existing_docs = corpus_store.list_documents(corpus_id)
    if max_files is not None and len(existing_docs) + len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Corpus is limited to {max_files} files")

    running_total = sum(d.get("size_bytes", 0) for d in existing_docs)
    pending_docs = []
    uploaded = []

    for position, upload in enumerate(files):
        safe_filename = sanitize_filename(upload.filename or "")
        extension = validate_extension(safe_filename)
        relative_path = sanitize_relative_path(
            relative_paths[position] if position < len(relative_paths) else None, safe_filename
        )
        content = await upload.read()
        size_bytes = len(content)

        if size_bytes == 0:
            raise HTTPException(status_code=400, detail=f"File is empty: {safe_filename}")
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds max size ({MAX_FILE_SIZE_MB} MB): {safe_filename}",
            )

        running_total += size_bytes
        if max_total_size_bytes is not None and running_total > max_total_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Corpus exceeds max total size ({max_total_size_bytes // (1024 * 1024)} MB)",
            )

        normalized_text = parse_document_bytes(content=content, extension=extension)
        if not normalized_text.strip():
            raise HTTPException(status_code=400, detail=f"No extractable text in file: {safe_filename}")

        document_id = str(uuid.uuid4())
        _, normalized_filename = make_normalized_filename(safe_filename, document_id)

        governance = infer_governance(normalized_text)
        pending_docs.append({
            "document_id": document_id,
            "original_filename": safe_filename,
            "normalized_filename": normalized_filename,
            "file_type": extension,
            "size_bytes": size_bytes,
            "normalized_text": normalized_text,
            "relative_path": relative_path,
            "original_bytes": content,
            "content_type": CONTENT_TYPES.get(extension, "application/octet-stream"),
            **governance,
        })
        uploaded.append({
            "document_id": document_id,
            "filename": relative_path,
            "original_filename": safe_filename,
            "relative_path": relative_path,
            "file_type": extension,
            "size_bytes": size_bytes,
            "status": "uploaded",
            **governance,
            "status_override": None,
            "doc_type_override": None,
        })

    # Single metadata.json write for the whole batch (GCS rate-limits per-object
    # mutations to ~1/sec, so writing once-per-file 429s on large batches).
    metadata = corpus_store.add_documents_bulk(corpus_id, pending_docs)

    # Uploading invalidates any cached GovRAGGemini instance for this corpus.
    invalidate_corpus_cache(corpus_id)

    return {
        "corpus_id": corpus_id,
        "uploaded": uploaded,
        "document_count": metadata.get("document_count", 0),
        "requires_reindex": True,
        "status": metadata.get("status"),
    }


@app.post("/corpora/{corpus_id}/documents")
async def upload_documents(
    corpus_id: str,
    files: List[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        return await handle_document_uploads(corpus_id, files)
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/corpora/{corpus_id}/documents")
async def list_documents(corpus_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        documents = corpus_store.list_documents(corpus_id)
        return {
            "corpus_id": corpus_id,
            "document_count": len(documents),
            "documents": documents,
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.delete("/corpora/{corpus_id}/documents/{document_id}")
async def delete_document(corpus_id: str, document_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        metadata = corpus_store.delete_document(corpus_id, document_id)
        invalidate_corpus_cache(corpus_id)
        return {
            "corpus_id": corpus_id,
            "document_id": document_id,
            "status": "deleted",
            "document_count": metadata.get("document_count", 0),
            "requires_reindex": True,
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/corpora/{corpus_id}/index")
async def index_corpus(
    corpus_id: str,
    payload: CorpusIndexRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        request = QueryRequest(
            question="index",
            project_id=payload.project_id,
            region=payload.region,
            use_llm=payload.use_llm,
            top_k=8,
        )
        _, cache_meta = ensure_corpus_rag(corpus_id, request, force_reindex=True)
        return {
            "corpus_id": corpus_id,
            "status": "ready",
            "document_count": cache_meta.get("document_count", 0),
            "message": "Corpus indexed successfully",
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/corpora/{corpus_id}/query", response_model=CorpusQueryResponse)
async def query_corpus(
    corpus_id: str,
    request: QueryRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)

    try:
        return run_corpus_query(corpus_id, request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"CORPUS QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# PUBLIC DEMO CORPUS API (for the Corvic browser Web App)
#
# No X-API-Key is required or accepted here. Safety instead relies on:
#   - unguessable corpus_id (uuid4, acts as a bearer capability)
#   - no endpoint that lists all demo corpora
#   - short TTL + opportunistic expiry cleanup
#   - file count/type/size caps and per-IP rate limiting
#   - server-controlled project_id/region (never taken from the client)
# These routes reuse the exact same corpus_store and GovRAGGemini pipeline
# as the protected /corpora endpoints; no RAG logic is duplicated here.
# ============================================================

@app.post("/demo/corpora")
async def create_demo_corpus(payload: DemoCorpusCreateRequest, http_request: Request):
    enforce_demo_rate_limit(http_request)

    # Best-effort cleanup of expired demo corpora; not a scheduled job.
    try:
        corpus_store.delete_expired_corpora(kind="demo")
    except Exception as exc:
        print(f"DEMO CLEANUP WARNING: {exc}")

    try:
        corpus_id = str(uuid.uuid4())
        metadata = corpus_store.create_corpus(
            corpus_id=corpus_id,
            name=payload.name,
            kind="demo",
            ttl_hours=DEMO_CORPUS_TTL_HOURS,
        )
        return {
            "corpus_id": metadata["corpus_id"],
            "name": metadata["name"],
            "status": metadata["status"],
            "document_count": metadata["document_count"],
            "created_at": metadata["created_at"],
            "expires_at": metadata["expires_at"],
        }
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/demo/corpora/{corpus_id}")
async def get_demo_corpus(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    metadata = get_demo_corpus_or_404(corpus_id)

    return {
        "corpus_id": metadata.get("corpus_id"),
        "name": metadata.get("name"),
        "status": metadata.get("status"),
        "document_count": metadata.get("document_count", 0),
        "created_at": metadata.get("created_at"),
        "expires_at": metadata.get("expires_at"),
    }


@app.post("/demo/corpora/{corpus_id}/documents")
async def upload_demo_documents(
    corpus_id: str,
    http_request: Request,
    files: List[UploadFile] = File(...),
    paths: Optional[List[str]] = Form(default=None),
):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        return await handle_document_uploads(
            corpus_id,
            files,
            max_files=MAX_DEMO_DOCUMENTS,
            max_total_size_bytes=MAX_DEMO_CORPUS_BYTES,
            relative_paths=paths,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


def resolve_source_folder(raw_path: str) -> str:
    """Resolve a user-typed folder path and check it sits under an allowed root."""
    if not ALLOWED_SOURCE_ROOTS:
        raise HTTPException(status_code=403, detail="The server folder connector is disabled (DOCUINSIGHT_ALLOWED_ROOTS is empty)")
    candidate = os.path.realpath(os.path.expanduser(raw_path.strip()))
    if not os.path.isdir(candidate):
        raise HTTPException(status_code=400, detail=f"Not a folder on the server: {raw_path}")
    if not any(candidate == root or candidate.startswith(root + os.sep) for root in ALLOWED_SOURCE_ROOTS):
        raise HTTPException(
            status_code=403,
            detail=f"Folder is outside the allowed roots ({', '.join(ALLOWED_SOURCE_ROOTS)})",
        )
    return candidate


def walk_source_folder(folder: str) -> List[str]:
    """Supported files under ``folder``, sorted; hidden entries and symlinks are skipped."""
    found: List[str] = []
    for current, dirnames, filenames in os.walk(folder, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and not os.path.islink(os.path.join(current, d)))
        for name in sorted(filenames):
            full = os.path.join(current, name)
            if name.startswith(".") or os.path.islink(full):
                continue
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                found.append(full)
    return found


def index_server_folder(
    corpus_id: str,
    raw_path: str,
    label: Optional[str] = None,
    max_files: Optional[int] = None,
    max_total_size_bytes: Optional[int] = None,
) -> Dict:
    """Index every supported file under a folder on the server, keeping its structure.

    Files that cannot be used (too large, empty, unparseable) are reported in
    ``skipped`` instead of failing the whole folder.
    """
    folder = resolve_source_folder(raw_path)
    paths = walk_source_folder(folder)
    if not paths:
        raise HTTPException(status_code=400, detail=f"No PDF, Word, text or Markdown files found under {raw_path}")

    source_label = sanitize_relative_path(label, "") or re_sub_unsafe(os.path.basename(folder.rstrip(os.sep))) or "folder"

    existing_docs = corpus_store.list_documents(corpus_id)
    if max_files is not None and len(existing_docs) + len(paths) > max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Folder holds {len(paths)} files; the collection is limited to {max_files} in total",
        )

    running_total = sum(d.get("size_bytes", 0) for d in existing_docs)
    pending_docs = []
    uploaded = []
    skipped = []

    for full_path in paths:
        rel = os.path.relpath(full_path, folder)
        relative_path = sanitize_relative_path(f"{source_label}/{rel}", os.path.basename(full_path))
        safe_filename = sanitize_filename(os.path.basename(full_path))
        try:
            extension = validate_extension(safe_filename)
            size_bytes = os.path.getsize(full_path)
            if size_bytes == 0:
                raise ValueError("file is empty")
            if size_bytes > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"larger than {MAX_FILE_SIZE_MB} MB")
            if max_total_size_bytes is not None and running_total + size_bytes > max_total_size_bytes:
                raise ValueError(f"collection would exceed {max_total_size_bytes // (1024 * 1024)} MB")
            with open(full_path, "rb") as handle:
                content = handle.read()
            normalized_text = parse_document_bytes(content=content, extension=extension)
            if not normalized_text.strip():
                raise ValueError("no extractable text")
        except HTTPException as exc:
            skipped.append({"path": rel, "reason": str(exc.detail)})
            continue
        except Exception as exc:
            skipped.append({"path": rel, "reason": str(exc)})
            continue

        running_total += size_bytes
        document_id = str(uuid.uuid4())
        _, normalized_filename = make_normalized_filename(safe_filename, document_id)
        governance = infer_governance(normalized_text)
        pending_docs.append({
            "document_id": document_id,
            "original_filename": safe_filename,
            "normalized_filename": normalized_filename,
            "file_type": extension,
            "size_bytes": size_bytes,
            "normalized_text": normalized_text,
            "relative_path": relative_path,
            "original_bytes": content,
            "content_type": CONTENT_TYPES.get(extension, "application/octet-stream"),
            **governance,
        })
        uploaded.append({
            "document_id": document_id,
            "filename": relative_path,
            "original_filename": safe_filename,
            "relative_path": relative_path,
            "file_type": extension,
            "size_bytes": size_bytes,
            "status": "uploaded",
            **governance,
            "status_override": None,
            "doc_type_override": None,
        })

    if not pending_docs:
        reasons = "; ".join(f"{s['path']}: {s['reason']}" for s in skipped[:5])
        raise HTTPException(status_code=400, detail=f"No usable files under {raw_path} ({reasons})")

    metadata = corpus_store.add_documents_bulk(corpus_id, pending_docs)
    invalidate_corpus_cache(corpus_id)

    return {
        "corpus_id": corpus_id,
        "source": {"type": "folder", "path": folder, "label": source_label, "files_found": len(paths)},
        "uploaded": uploaded,
        "skipped": skipped,
        "document_count": metadata.get("document_count", 0),
        "requires_reindex": True,
        "status": metadata.get("status"),
    }


@app.post("/corpora/{corpus_id}/sources/folder")
async def add_folder_source(
    corpus_id: str,
    payload: FolderSourceRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key)
    validate_corpus_id(corpus_id)
    try:
        return index_server_folder(corpus_id, payload.path, payload.label)
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/demo/corpora/{corpus_id}/sources/folder")
async def add_demo_folder_source(corpus_id: str, payload: FolderSourceRequest, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)
    try:
        return index_server_folder(
            corpus_id,
            payload.path,
            payload.label,
            max_files=MAX_DEMO_DOCUMENTS,
            max_total_size_bytes=MAX_DEMO_CORPUS_BYTES,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


def load_example_documents(corpus_id: str) -> Dict:
    """One-click load of the bundled 180-document Easy pack into a demo corpus."""
    paths = sorted(glob.glob(os.path.join(DEMO_DOCUMENTS_DIR, "*.md")))
    if not paths:
        raise HTTPException(status_code=500, detail="Example dataset not available")

    existing_docs = corpus_store.list_documents(corpus_id)
    if len(existing_docs) + len(paths) > MAX_DEMO_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Corpus is limited to {MAX_DEMO_DOCUMENTS} files")

    running_total = sum(d.get("size_bytes", 0) for d in existing_docs)
    pending_docs = []
    uploaded = []

    for path in paths:
        text = open(path, "r", encoding="utf-8").read()
        size_bytes = len(text.encode("utf-8"))

        running_total += size_bytes
        if running_total > MAX_DEMO_CORPUS_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Corpus exceeds max total size ({MAX_DEMO_CORPUS_MB} MB)",
            )

        original_filename = os.path.basename(path)
        document_id = str(uuid.uuid4())
        _, normalized_filename = make_normalized_filename(original_filename, document_id)

        governance = infer_governance(text)
        pending_docs.append({
            "document_id": document_id,
            "original_filename": original_filename,
            "normalized_filename": normalized_filename,
            "file_type": ".md",
            "size_bytes": size_bytes,
            "normalized_text": text,
            "relative_path": f"example/{original_filename}",
            "original_bytes": text.encode("utf-8"),
            "content_type": CONTENT_TYPES[".md"],
            **governance,
        })
        uploaded.append({
            "document_id": document_id,
            "filename": f"example/{original_filename}",
            "original_filename": original_filename,
            "relative_path": f"example/{original_filename}",
            "file_type": ".md",
            "size_bytes": size_bytes,
            "status": "uploaded",
            **governance,
            "status_override": None,
            "doc_type_override": None,
        })

    # Single metadata.json write for all 180 files (see handle_document_uploads).
    metadata = corpus_store.add_documents_bulk(corpus_id, pending_docs)
    invalidate_corpus_cache(corpus_id)

    return {
        "corpus_id": corpus_id,
        "uploaded": uploaded,
        "document_count": metadata.get("document_count", 0),
        "requires_reindex": True,
        "status": metadata.get("status"),
    }


@app.post("/demo/corpora/{corpus_id}/load_example")
async def load_example_dataset(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        return load_example_documents(corpus_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/demo/corpora/{corpus_id}/documents")
async def list_demo_documents(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        documents = corpus_store.list_documents(corpus_id)
        return {
            "corpus_id": corpus_id,
            "document_count": len(documents),
            "documents": [public_document_view(d) for d in documents],
        }
    except Exception as exc:
        raise map_store_error(exc)


def public_document_view(d: Dict) -> Dict:
    return {
        "document_id": d.get("document_id"),
        "filename": d.get("relative_path") or d.get("original_filename"),
        "original_filename": d.get("original_filename"),
        "relative_path": d.get("relative_path") or d.get("original_filename"),
        "file_type": d.get("file_type"),
        "uploaded_at": d.get("uploaded_at"),
        "size_bytes": d.get("size_bytes"),
        "inferred_status": d.get("inferred_status", "unknown"),
        "inferred_doc_type": d.get("inferred_doc_type", "unknown"),
        "status_override": d.get("status_override"),
        "doc_type_override": d.get("doc_type_override"),
        "has_original": bool(d.get("original_path")),
    }


@app.patch("/demo/corpora/{corpus_id}/documents/{document_id}")
async def override_demo_document(corpus_id: str, document_id: str, payload: DocumentOverrideRequest, http_request: Request):
    """Set or clear the user's governance overrides for one document."""
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)
    if payload.status_override is not None and payload.status_override not in STATUS_OPTIONS:
        raise HTTPException(status_code=400, detail=f"status_override must be one of {STATUS_OPTIONS}")
    if payload.doc_type_override is not None and payload.doc_type_override not in DOC_TYPE_OPTIONS:
        raise HTTPException(status_code=400, detail=f"doc_type_override must be one of {DOC_TYPE_OPTIONS}")
    fields: Dict = {}
    if payload.clear:
        fields = {"status_override": None, "doc_type_override": None}
    else:
        if "status_override" in payload.model_fields_set:
            fields["status_override"] = payload.status_override
        if "doc_type_override" in payload.model_fields_set:
            fields["doc_type_override"] = payload.doc_type_override
    if not fields:
        raise HTTPException(status_code=400, detail="Nothing to update")
    try:
        doc = corpus_store.update_document(corpus_id, document_id, fields)
        invalidate_corpus_cache(corpus_id)
        return public_document_view(doc)
    except Exception as exc:
        raise map_store_error(exc)


@app.delete("/demo/corpora/{corpus_id}/documents/{document_id}")
async def delete_demo_document(corpus_id: str, document_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)
    try:
        metadata = corpus_store.delete_document(corpus_id, document_id)
        invalidate_corpus_cache(corpus_id)
        document_text_cache.pop(f"{corpus_id}|{document_id}", None)
        return {"corpus_id": corpus_id, "document_id": document_id, "status": "deleted",
                "document_count": metadata.get("document_count", 0)}
    except Exception as exc:
        raise map_store_error(exc)


@app.get("/demo/corpora/{corpus_id}/documents/{document_id}/text")
async def demo_document_text(corpus_id: str, document_id: str, http_request: Request):
    """Full normalized text plus chunk spans, for the reader pane."""
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)
    try:
        doc = corpus_store.get_document(corpus_id, document_id)
        text = get_document_text(corpus_id, document_id) or ""
    except Exception as exc:
        raise map_store_error(exc)

    chunks: List[Dict] = []
    with cache_lock:
        instances = [rag for key, rag in rag_instances.items() if key.startswith(f"{corpus_id}|")]
    if instances:
        rag = instances[0]
        chunks = [
            {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "text": c.text,
             "page_start": c.page_start, "page_end": c.page_end, "section": c.section}
            for c in rag.chunks if c.document_id == document_id
        ]
        chunks = [{k: v for k, v in c.items() if k != "text"} for c in locate_chunks(chunks, text)]
    pages = []
    if "\f" in text:
        offset = 0
        for number, page_text in enumerate(text.split("\f"), start=1):
            pages.append({"page": number, "start": offset, "end": offset + len(page_text)})
            offset += len(page_text) + 1
    return {
        **public_document_view(doc),
        "text": text,
        "chunks": chunks,
        "pages": pages,
    }


@app.get("/demo/corpora/{corpus_id}/documents/{document_id}/original")
async def demo_document_original(corpus_id: str, document_id: str, http_request: Request, format: Optional[str] = None):
    """The file as uploaded (PDF, Markdown, text) or, for Word files with
    ``?format=html``, an HTML rendering for in-browser display."""
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)
    try:
        doc = corpus_store.get_document(corpus_id, document_id)
        data = corpus_store.read_original(corpus_id, document_id)
    except Exception as exc:
        raise map_store_error(exc)
    file_type = doc.get("file_type", "")
    if file_type == ".docx" and format == "html":
        try:
            import mammoth
        except ImportError as exc:
            raise HTTPException(status_code=501, detail="Word rendering needs the 'mammoth' package") from exc
        html = mammoth.convert_to_html(io.BytesIO(data)).value
        return HTMLResponse(content=html)
    media_type = CONTENT_TYPES.get(file_type, "application/octet-stream")
    headers = {"Content-Disposition": f"inline; filename=\"{doc.get('original_filename', 'document')}\""}
    return Response(content=data, media_type=media_type, headers=headers)


@app.delete("/demo/corpora/{corpus_id}")
async def delete_demo_corpus(corpus_id: str, http_request: Request):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    try:
        corpus_store.delete_corpus(corpus_id)
        invalidate_corpus_cache(corpus_id)
        return {"corpus_id": corpus_id, "status": "deleted"}
    except Exception as exc:
        raise map_store_error(exc)


@app.post("/demo/corpora/{corpus_id}/query", response_model=CorpusQueryResponse)
async def query_demo_corpus(
    corpus_id: str,
    payload: DemoQueryRequest,
    http_request: Request,
):
    enforce_demo_rate_limit(http_request)
    get_demo_corpus_or_404(corpus_id)

    # project_id/region/use_llm are fixed server-side; never taken from the browser.
    request = QueryRequest(
        question=payload.question,
        project_id=DEMO_PROJECT_ID,
        region=DEMO_REGION,
        use_llm=True,
        top_k=payload.top_k,
        context=payload.context,
    )

    try:
        return run_corpus_query(corpus_id, request)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"DEMO CORPUS QUERY ERROR: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
