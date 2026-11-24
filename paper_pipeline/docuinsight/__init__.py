"""DocuInsight: conflict-aware retrieval-augmented answering over documents that disagree.

This package is the system configuration used in the paper ("Variant 3"):

* first-stage retrieval: BM25 + dense fusion (``retrieval="hybrid"``)
* governance-aware reranking and conflict diversification (unchanged core)
* resolution policy: answer-and-flag (``policy="answer_and_flag"``), i.e.
  governance -> corroboration by distinct documents -> consistency -> tie.

The refuse-on-conflict ablation is ``policy="strict"``; the dense-retrieval
ablation is ``retrieval="dense"``. Both remain available as arguments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .core import Document, GovRAGGemini
from .document_parser import SUPPORTED_EXTENSIONS, parse_document_bytes
from .providers import LocalEmbedder, OpenAICompatGenerator

__all__ = [
    "Document",
    "GovRAGGemini",
    "LocalEmbedder",
    "OpenAICompatGenerator",
    "DEFAULT_RETRIEVAL",
    "DEFAULT_POLICY",
    "load_folder",
    "build_pipeline",
]

DEFAULT_RETRIEVAL = "hybrid"
DEFAULT_POLICY = "answer_and_flag"


def load_folder(folder: str) -> List[Document]:
    """Load every supported file (.md, .txt, .pdf, .docx) under ``folder``.

    The document name is the path relative to ``folder`` so a folder tree keeps
    its structure in citations.
    """
    base = Path(folder)
    if not base.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    docs: List[Document] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        text = parse_document_bytes(path.read_bytes(), ext)
        if text and text.strip():
            docs.append(Document(filename=str(path.relative_to(base)), text=text))
    if not docs:
        raise ValueError(f"No supported documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found under {folder}")
    return docs


def build_pipeline(
    documents: List[Document],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embed_model: Optional[str] = None,
    use_llm: bool = True,
) -> GovRAGGemini:
    """Index ``documents`` and return a ready pipeline.

    ``model`` and ``base_url`` select any OpenAI-compatible chat endpoint
    (defaults: ``NEMOTRON_MODEL`` / ``DOCUINSIGHT_BASE_URL`` env vars, then the
    NVIDIA hosted endpoint). The embedder runs locally.
    """
    embedder = LocalEmbedder(model_name=embed_model) if embed_model else LocalEmbedder()
    generator = OpenAICompatGenerator(model=model, base_url=base_url, api_key=api_key) if use_llm else None
    return GovRAGGemini(None, embedder=embedder, generator=generator, use_llm=use_llm, documents=documents)
