# DocuInsight / GOV-RAG

FastAPI service for authority-aware conflicting-document retrieval and resolution.

The application supports two modes:

- Fixed demo corpus: bundled `easy/` (existing `/query` and `/demo_query` behavior)
- Dynamic corpora: create/upload/index/query your own collections under `/corpora/*`

## Architecture

Flow:

Corvic or client
-> GOV-RAG FastAPI
-> corpus documents (GCS or local fallback)
-> semantic retrieval
-> authority + scope + status reranking
-> conflict diversification
-> Gemini structured decision and citation validation
-> answer, verified chunk citations, competing evidence, and decision explanation

Modules:

- `main.py`: routes, auth, per-corpus RAG cache
- `pipeline/corpus_store.py`: corpus metadata + document persistence (GCS/local)
- `pipeline/document_parser.py`: normalization for `.md`, `.txt`, `.pdf`, `.docx`
- `pipeline/gov_rag_gemini.py`: existing GOV-RAG core ranking/reasoning
- `pipeline/chunker.py`: overlapping chunk creation for long documents
- `pipeline/rate_limit.py`: in-memory public demo API rate limiting

## Evidence and decisions

Responses keep the original legacy fields (`answer`, `selected_source`, `reason`,
`conflict_detected`, `confidence`, `num_conflicts`, and `top_sources`) and add:

- `decision`: `answered`, `unresolved_conflict`, `needs_information`, or
  `insufficient_evidence`
- `citations`: verified chunk IDs, exact quoted text, source locations, and
  source treatment (`governing`, `supporting`, `superseded`, `different_scope`,
  `conflicting`, or `uncertain`)
- `explanation`, `missing_information`, and `clarification_question`
- Chunk-level `top_sources`, including full passage, stable document/chunk IDs,
  original filename, page/section when available, inferred status/type, and
  retrieval component scores. Multiple chunks from the same document are kept.

Numeric `claim` and `supporting_sentence` values are auxiliary retrieval
metadata, not citations to the generated answer. The API validates every
returned citation ID and quote against the retrieved passages. Invalid output
is returned as a visible `insufficient_evidence` result without a selected
source. `selected_source` remains nullable for legacy clients and unresolved
conflicts.

Status, type, and authority are text-inferred signals, not verified
organizational facts. Scope score is a similarity measure, not proof that a
source applies. Evidence is not silently discarded just because it comes from
the same document, and unresolved sources are not assigned an invented winner.

PDF page text is stored with explicit page boundaries so page locations survive
chunking and persisted indexes. Markdown headings and DOCX heading styles are
used as section labels where available; other chunks still have stable chunk
references. Persisted indexes use format version 3. Older versions do not have
reliable location metadata and are rejected for controlled reindexing on the
next query/index operation.

Query bodies remain compatible with question-only clients. Optional case facts
can be supplied in `context`:

```json
{
  "question": "Which leave rule applies?",
  "context": {
    "relevant_date": "2026-10-31",
    "location": "Ontario",
    "employee_category": "hourly",
    "circumstances": "The employee is returning from approved leave."
  }
}
```

## Persistence Model

Cloud Run file system is ephemeral. Persistent corpus storage uses GCS when configured.

Set:

```bash
export GOVRAG_BUCKET="your-bucket-name"
```

GCS layout:

```text
corpora/
  <corpus_id>/
    metadata.json
    documents/
      <normalized_document>.md
```

Local fallback (when `GOVRAG_BUCKET` is unset):

```text
./corpora/<corpus_id>/
```

## Environment Variables

- `GOOGLE_CLOUD_PROJECT` (required for Vertex AI)
- `VERTEX_AI_REGION` (default `us-central1`)
- `GOVRAG_API_SECRET` (optional; if set, protected routes require `X-API-Key`)
- `GOVRAG_BUCKET` (optional; enables persistent GCS corpus storage)
- `GOVRAG_MAX_FILE_SIZE_MB` (optional; default `20`)
- `GOVRAG_ALLOWED_ORIGINS` (optional; comma-separated CORS origins for the browser demo API, default `https://app.corvic.ai`)
- `DEMO_CORPUS_TTL_HOURS` (optional; default `24` — demo corpora auto-expire)
- `GOVRAG_MAX_DEMO_DOCUMENTS` (optional; default `500`)
- `GOVRAG_MAX_DEMO_CORPUS_MB` (optional; default `500`)
- `GOVRAG_MAX_QUESTION_LENGTH` (optional; default `2000`)
- `GOVRAG_DEMO_RATE_LIMIT_PER_MINUTE` (optional; default `30`, per-IP, in-memory)

## Local Run

```bash
cd corvic
pip install -r requirements.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="project-79920195-9e86-44ea-8c9"
export VERTEX_AI_REGION="us-central1"
export GOVRAG_API_SECRET="$(openssl rand -hex 32)"
# Optional for GCS persistence
# export GOVRAG_BUCKET="your-bucket"

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

or:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app). The first query
for an unindexed collection creates embeddings; answer generation also calls
Gemini when enabled. Those live query operations can incur cloud charges.
Offline tests below mock embeddings and generation and do not call Google
services.

## Endpoints

Public:

- `GET /`
- `GET /health`
- `POST /demo_query` (demo corpus, no API key)

Protected (when `GOVRAG_API_SECRET` is set):

- `POST /query` (existing easy corpus)
- `POST /batch_query` (existing easy corpus)
- `POST /corpora`
- `GET /corpora`
- `GET /corpora/{corpus_id}`
- `POST /corpora/{corpus_id}/documents` (multipart, multiple files)
- `GET /corpora/{corpus_id}/documents`
- `DELETE /corpora/{corpus_id}/documents/{document_id}`
- `POST /corpora/{corpus_id}/index`
- `POST /corpora/{corpus_id}/query`

Supported upload file types:

- `.md`
- `.txt`
- `.pdf`
- `.docx`

## Public Browser Demo API (DocuInsight)

The Corvic browser app cannot securely hold `GOVRAG_API_SECRET`, so a separate
**unauthenticated** set of routes exists specifically for short-lived, browser-driven
sessions. These reuse the exact same `corpus_store` and `GovRAGGemini` pipeline as the
protected `/corpora` API — no logic is duplicated.

```
POST   /demo/corpora                      -> create a session corpus, returns corpus_id
POST   /demo/corpora/{corpus_id}/documents -> upload files (multipart, multiple)
POST   /demo/corpora/{corpus_id}/query     -> auto-indexes if needed, returns GOV-RAG result
GET    /demo/corpora/{corpus_id}
GET    /demo/corpora/{corpus_id}/documents
DELETE /demo/corpora/{corpus_id}
```

Security properties:

- No `X-API-Key` required or accepted; `project_id`/`region` are fixed server-side
  (`GOOGLE_CLOUD_PROJECT`/`VERTEX_AI_REGION`) and are never taken from the client.
- `corpus_id` is a cryptographically random UUID and acts as the only access
  credential — there is intentionally **no** `GET /demo/corpora` listing endpoint.
- Demo corpora expire after `DEMO_CORPUS_TTL_HOURS` (default 24h); expired corpora are
  cleaned up opportunistically when a new demo corpus is created, and are treated as
  `404 Not Found` even if not yet swept.
- Uploads are capped at `GOVRAG_MAX_DEMO_DOCUMENTS` files,
  `GOVRAG_MAX_FILE_SIZE_MB` per file, and `GOVRAG_MAX_DEMO_CORPUS_MB` per corpus.
- Questions are capped at `GOVRAG_MAX_QUESTION_LENGTH` characters.
- Per-IP in-memory rate limiting (`GOVRAG_DEMO_RATE_LIMIT_PER_MINUTE`) applies to all
  `/demo/corpora/*` routes. This is single-process and resets on redeploy — fine for a
  demo, not a substitute for a real gateway if you scale beyond one instance.
- CORS is restricted to `GOVRAG_ALLOWED_ORIGINS` (default `https://app.corvic.ai`);
  `allow_credentials` is never combined with a wildcard origin.
- No Gemini/GCP credentials or `GOVRAG_API_SECRET` are ever present in any response.

Browser workflow:

```bash
# 1. Create a workspace
curl -sS -X POST "$URL/demo/corpora" -H "Content-Type: application/json" -d '{"name":"My Workspace"}'
# -> {"corpus_id": "...", "expires_at": "..."}

# 2. Upload files
curl -sS -X POST "$URL/demo/corpora/$CORPUS_ID/documents" \
  -F "files=@official_policy.md" -F "files=@draft_policy.md"

# 3. Ask a question (auto-indexes on first query, no explicit /index call needed)
curl -sS -X POST "$URL/demo/corpora/$CORPUS_ID/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What defect rate currently applies to Q4-certified suppliers?"}'
```

The response gives the DocuInsight UI the legacy fields plus a structured
decision, verified citations, full passages, missing case information, and
source treatment details.

UI-only states can be previewed without a collection or API call by opening
`/app?fixture=answered`, `/app?fixture=unresolved_conflict`,
`/app?fixture=needs_information`, or `/app?fixture=insufficient_evidence`. Each
preview is explicitly labeled as a local UI fixture and is not part of the
answer path.

## cURL Examples

Set base vars:

```bash
URL="http://localhost:8000"
KEY="your-api-secret"
```

Create corpus:

```bash
curl -sS -X POST "$URL/corpora" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"name":"Meridian Supplier Policies"}'
```

Upload files:

```bash
CORPUS_ID="<from create response>"
curl -sS -X POST "$URL/corpora/$CORPUS_ID/documents" \
  -H "X-API-Key: $KEY" \
  -F "files=@official_policy.md" \
  -F "files=@draft_policy.md" \
  -F "files=@internal_summary.md"
```

Index corpus:

```bash
curl -sS -X POST "$URL/corpora/$CORPUS_ID/index" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{
    "project_id": "project-79920195-9e86-44ea-8c9",
    "region": "us-central1",
    "use_llm": true
  }'
```

Query corpus:

```bash
curl -sS -X POST "$URL/corpora/$CORPUS_ID/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{
    "question": "What defect rate currently applies to Q4-certified suppliers?",
    "project_id": "project-79920195-9e86-44ea-8c9",
    "region": "us-central1",
    "use_llm": true,
    "top_k": 8
  }'
```

## Test Scripts

Focused, offline tests for structured decisions, citation verification,
locations, and index-version handling:

```bash
python -m pytest -q test_inspectable_evidence.py
```

These tests do not invoke Gemini, Vertex AI, or any live endpoint. Existing
workflow scripts below may perform live inference; run them only when that is
intended.

API smoke test:

```bash
export GOVRAG_API_KEY="$GOVRAG_API_SECRET"
python test_api.py
```

Dynamic corpus workflow test:

```bash
export GOVRAG_API_KEY="$GOVRAG_API_SECRET"
python test_corpus_workflow.py --use-llm
```

Public browser demo API test (no API key):

```bash
python test_demo_workflow.py
```

`--skip-demo-query` skips that script's demo answer call, but other workflow
steps can still build embeddings and reach Google services; it is not an
offline test:

```bash
python test_corpus_workflow.py --skip-demo-query
```

## Cloud Run Deployment

```bash
cd ~/Documents/Conflicting_documents/corvic
gcloud config set project project-79920195-9e86-44ea-8c9

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com

SECRET="$(openssl rand -hex 32)"

gcloud run deploy govrag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 --timeout 600 --concurrency 4 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=project-79920195-9e86-44ea-8c9,VERTEX_AI_REGION=us-central1,GOVRAG_API_SECRET=${SECRET},GOVRAG_BUCKET=your-bucket-name,GOVRAG_ALLOWED_ORIGINS=https://app.corvic.ai,DEMO_CORPUS_TTL_HOURS=24"
```

If you want to keep one instance warm for latency:

```bash
gcloud run services update govrag-api --region us-central1 --min-instances=1
```
