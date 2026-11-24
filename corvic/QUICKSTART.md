# DocuInsight Quick Start

## Run locally

```bash
cd corvic
python -m pip install -r requirements.txt
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export VERTEX_AI_REGION="us-central1"
gcloud auth application-default login
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app). Upload a collection
or load the example dataset, enter a question, and optionally add case context.
The first query may create a persisted embedding index; Gemini generation and
embedding operations can incur cloud charges.

To preview UI states without creating a collection or calling the API, use
`/app?fixture=answered`, `/app?fixture=unresolved_conflict`,
`/app?fixture=needs_information`, or `/app?fixture=insufficient_evidence`. The
page labels these as local UI fixtures.

## Offline tests

```bash
python -m pytest -q test_inspectable_evidence.py
```

This focused suite uses mocked generation and embeddings and makes no live
Google Cloud or Gemini calls. Do not run workflow/evaluation scripts when
avoiding paid inference.

## Compatibility API

Existing clients can continue sending only a question:

```bash
curl -X POST http://127.0.0.1:8000/demo_query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the policy say?"}'
```

To provide optional case details, send `context` with any of `relevant_date`,
`location`, `employee_category`, or `circumstances`. The response keeps legacy
fields and adds a decision, verified chunk citations, full passages, and any
missing information or clarification question.
