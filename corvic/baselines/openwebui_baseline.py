"""Open WebUI document-chat baseline: the stock product, driven over its REST API.

Open WebUI (v0.13.4) runs locally as the real application (see
~/.open-webui-eval/serve.sh), with every retrieval setting left at its shipped
default: sentence-transformers/all-MiniLM-L6-v2 embeddings, recursive
character splitting at 1024 characters with 100 overlap, top-k 10, no hybrid
search, no reranker, relevance threshold 0, the stock RAG prompt template, and
retrieval query generation on. Only the LLM endpoint is changed, to the same
NVIDIA-hosted generator used by every other row.

Per question: upload each document as a text file (synchronous processing),
ask one stateless chat completion that references those files, read the
answer plus the retrieved sources, then delete the files. Uploads are
serialised because the server's embedder runs on MPS, which is not
thread-safe; completions run concurrently.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from pipeline.providers import DEFAULT_NVIDIA_MODEL

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
DEFAULT_URL = os.environ.get("OPENWEBUI_URL", "http://127.0.0.1:8089")
ADMIN_ENV = Path.home() / ".open-webui-eval" / "admin.env"


def _read_admin_env() -> dict:
    out = {}
    if ADMIN_ENV.exists():
        for line in ADMIN_ENV.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


class OpenWebUIBaseline:
    def __init__(self, model: Optional[str] = None, base_url: str = DEFAULT_URL, timeout: float = 600.0):
        self.model = model or os.getenv("NEMOTRON_MODEL") or DEFAULT_NVIDIA_MODEL
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.upload_lock = threading.Lock()
        self.session = requests.Session()
        env = _read_admin_env()
        email = os.getenv("OPENWEBUI_EMAIL") or env.get("WEBUI_ADMIN_EMAIL")
        password = os.getenv("OPENWEBUI_PASSWORD") or env.get("WEBUI_ADMIN_PASSWORD")
        if not (email and password):
            raise RuntimeError(f"Open WebUI admin credentials not found ({ADMIN_ENV})")
        r = self.session.post(f"{self.base}/api/v1/auths/signin", json={"email": email, "password": password}, timeout=60)
        r.raise_for_status()
        self.session.headers["Authorization"] = f"Bearer {r.json()['token']}"
        cfg = self.session.get(f"{self.base}/api/v1/retrieval/config", timeout=60).json()
        keep = {k: cfg.get(k) for k in ("CHUNK_SIZE", "CHUNK_OVERLAP", "TEXT_SPLITTER", "ENABLE_MARKDOWN_HEADER_TEXT_SPLITTER",
                                        "TOP_K", "TOP_K_RERANKER", "RELEVANCE_THRESHOLD", "ENABLE_RAG_HYBRID_SEARCH",
                                        "HYBRID_BM25_WEIGHT", "RAG_RERANKING_ENGINE", "RAG_RERANKING_MODEL",
                                        "RAG_FULL_CONTEXT", "BYPASS_EMBEDDING_AND_RETRIEVAL")}
        emb = self.session.get(f"{self.base}/api/v1/retrieval/embedding", timeout=60).json()
        self.config = {"embedding_engine": emb.get("RAG_EMBEDDING_ENGINE"), "embedding_model": emb.get("RAG_EMBEDDING_MODEL"),
                       **keep, "rag_template_is_stock": (cfg.get("RAG_TEMPLATE") or "").startswith("### Task:")}
        print("✓ Open WebUI baseline (stock product over REST)")
        print(f"  {self.base}  model={self.model}")
        print(f"  retrieval config: {json.dumps(self.config)}")

    # ------------------------------------------------------------------ helpers
    def _upload(self, name: str, text: str) -> Optional[str]:
        files = {"file": (name if name.endswith(".txt") else f"{name}.txt", text.encode("utf-8"), "text/plain")}
        r = self.session.post(f"{self.base}/api/v1/files/", params={"process": "true", "process_in_background": "false"},
                              files=files, timeout=self.timeout)
        if r.status_code >= 400:
            if "DUPLICATE" in r.text.upper():
                return None
            r.raise_for_status()
        return r.json()["id"]

    def _delete(self, file_id: str) -> None:
        try:
            self.session.delete(f"{self.base}/api/v1/files/{file_id}", timeout=60)
        except Exception:  # noqa: BLE001 - cleanup only
            pass

    # -------------------------------------------------------------------- query
    def query(self, question: str, documents) -> dict:
        started = time.perf_counter()
        file_ids, id_to_name, skipped = [], {}, []
        with self.upload_lock:
            for d in documents:
                if not (d.text or "").strip():
                    continue
                fid = self._upload(d.filename, d.text)
                if fid is None:
                    skipped.append(d.filename)
                    continue
                file_ids.append(fid)
                id_to_name[fid] = d.filename
        error, answer, sources = None, "", []
        transport_failed = False
        try:
            body = {"model": self.model, "stream": False,
                    "messages": [{"role": "user", "content": question}],
                    "files": [{"type": "file", "id": fid} for fid in file_ids]}
            # Open WebUI surfaces upstream 429s/5xx from the generation endpoint as
            # HTTP errors; retry them with the same patient schedule as the other
            # providers instead of recording a spurious empty answer.
            data = None
            for attempt in range(10):
                try:
                    r = self.session.post(f"{self.base}/api/chat/completions", json=body, timeout=self.timeout)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(f"{r.status_code} {r.text[:200]}", response=r)
                    r.raise_for_status()
                    data = r.json()
                    break
                except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                    resp = getattr(exc, "response", None)
                    retryable = resp is None or resp.status_code in (429, 500, 502, 503, 504)
                    if not retryable or attempt == 9:
                        raise
                    time.sleep(min(60.0, 2.0 * (2 ** attempt)))
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            answer = _THINK_BLOCK.sub("", content).strip()
            sources = data.get("sources") or []
        except Exception as exc:  # noqa: BLE001 - recorded per item
            error = f"{type(exc).__name__}: {exc}"
            transport_failed = True
        finally:
            for fid in file_ids:
                self._delete(fid)
        candidates = []
        for s in sources:
            src = s.get("source") or {}
            fid = src.get("id") or (src.get("file") or {}).get("id")
            fname = id_to_name.get(fid) or src.get("name") or (src.get("file") or {}).get("filename")
            if fname and fname.endswith(".txt") and fname[:-4] in id_to_name.values():
                fname = fname[:-4]
            dists = s.get("distances") or []
            for i, chunk in enumerate(s.get("document") or []):
                candidates.append({"filename": fname, "chunk_id": f"{fname}#{i}",
                                   "semantic_score": dists[i] if i < len(dists) else None,
                                   "final_score": dists[i] if i < len(dists) else None, "text": chunk})
        return {
            "question": question,
            "decision": "free_text",
            "answer": answer,
            "short_answers": [],
            "citations": [],
            "candidates": candidates,
            "source": None,
            "selected_source": None,
            # The harness treats this exact reason as a transport failure and leaves the
            # item out of the checkpoint so a resumed run retries it.
            "reason": ("Generation failed; no answer was selected." if transport_failed else None)
                      or error or (f"skipped duplicate uploads: {skipped}" if skipped else None),
            "explanation": None,
            "confidence": None,
            "validation_error": bool(error),
            "repair_attempts": 0,
            "dropped_citations": 0,
            "retrieval_mode": "openwebui_stock",
            "_latency_s": time.perf_counter() - started,
        }
