#!/usr/bin/env python3
"""
GOV-RAG with Gemini Embeddings

This version uses Google's Gemini API for embeddings instead of 
local sentence-transformers. Useful when:
- You have network/proxy issues downloading models
- You want to use GCP infrastructure
- You prefer cloud-based embeddings

All other components (authority detection, conflict resolution) remain the same.
"""

import os
import re
import json
import glob
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .chunker import chunk_text_with_locations

# The Google SDKs are optional: the packaged pipeline defaults to a local
# embedder and an OpenAI-compatible generator (see ``providers``). The Gemini
# classes below raise a clear error only if they are actually constructed.
try:
    from vertexai.language_models import TextEmbeddingModel
    import vertexai
    USE_VERTEX_AI = True
except ImportError:
    USE_VERTEX_AI = False
    vertexai = None
    TextEmbeddingModel = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================
# CONFIG
# ============================================================

# Vertex AI Configuration (project-based authentication)
VERTEX_AI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", None)
VERTEX_AI_REGION = os.getenv("VERTEX_AI_REGION", "us-central1")
VERTEX_AI_MODEL = "text-embedding-004"  # or "textembedding-gecko@003"
VERTEX_AI_GENERATION_MODEL = "gemini-2.5-pro"  # Gemini 2.5 Pro

# Fallback: AI Studio API Key (if not using Vertex AI)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# ============================================================
# SYSTEM PROMPT FOR FINAL EVIDENCE RESOLUTION
# ============================================================

SYSTEM_PROMPT = """You answer questions using only the supplied evidence passages and optional case context.
Read full passages, including conditions and exceptions. Never infer organizational authority
as verified fact: status, document type, authority scores, and scope similarity are inferred
signals only; scope similarity is not proof of applicability. Never invent precedence.

Choose exactly one decision:
- answered: evidence supports a direct answer, including its conditions and exceptions.
- unresolved_conflict: applicable evidence disagrees and documented precedence does not resolve it.
- needs_information: relevant rules exist, but essential facts about this case are missing.
- insufficient_evidence: retrieved passages do not support an answer.

Do not force a winner between equally applicable conflicting passages. Distinguish different
scope, explicitly superseded material, supporting context, and uncertain applicability. A
document saying it "supersedes" earlier documents is not itself superseded.

Return JSON only:
{
  "decision": "answered|unresolved_conflict|needs_information|insufficient_evidence",
  "answer": "Readable answer with all important conditions and exceptions.",
  "short_answers": ["each distinct final answer as a short span; for an unresolved conflict, each competing answer"],
  "citations": [
    {"chunk_id": "exact supplied chunk id", "quote": "verbatim excerpt from that passage",
     "treatment": "governing|supporting|superseded|different_scope|conflicting|uncertain",
     "explanation": "brief plain-language treatment"}
  ],
  "explanation": "Brief explanation of this decision.",
  "missing_information": [],
  "clarification_question": null,
  "selected_source": null,
  "conflict_detected": false,
  "confidence": "high|medium|low"
}
For unresolved conflicts, cite each materially competing passage and explain its treatment;
selected_source must be null. For needs_information, list the missing facts and ask a useful
clarification question. Use an empty citation list if evidence cannot support any citation.
When the decision is answered, selected_source must be the filename of the governing passage the
answer is taken from (never null). selected_source must exactly match a supplied filename and
appear in citations. Numeric claim metadata is a search aid, never a citation or proof."""

# GOV-RAG "answer-and-flag" conflict policy. Same governance reasoning as
# SYSTEM_PROMPT, but when applicable passages disagree and no governance signal
# (status, supersession, scope) separates them, the model commits to the
# better-corroborated value and flags the dissent instead of abstaining.
# ``unresolved_conflict`` is reserved for genuine ties.
SYSTEM_PROMPT_ANSWER_AND_FLAG = """You answer questions using only the supplied evidence passages and optional case context.
Read full passages, including conditions and exceptions. Never infer organizational authority
as verified fact: status, document type, authority scores, and scope similarity are inferred
signals only; scope similarity is not proof of applicability. Never invent precedence.

Choose exactly one decision:
- answered: evidence supports a direct answer, including its conditions and exceptions.
- unresolved_conflict: applicable passages disagree AND the competing values are equally
  supported (see the tie rule below).
- needs_information: relevant rules exist, but essential facts about this case are missing.
- insufficient_evidence: retrieved passages do not support an answer.

Resolving disagreement, in order:
1. Governance: a passage that is explicitly superseded, out of scope, or supporting context
   does not compete. A document saying it "supersedes" earlier documents is not itself superseded.
2. Corroboration: among the passages that do compete, prefer the value supported by more
   independent documents (different filenames). Treat the "Corroboration" summary, when
   supplied, as a count aid only; verify it against the passages.
3. Consistency: prefer the value that agrees with the other facts in the evidence (dates,
   places, names, totals) and that comes from a more specific, internally consistent passage.
4. Tie rule: only if the competing values are supported by the same number of independent
   documents and steps 1 and 3 do not separate them, choose unresolved_conflict.

When you commit under step 2 or 3, the decision is answered, conflict_detected is true, the
dissenting passage is cited with treatment "conflicting", and the explanation names the
rejected value and why it lost. short_answers must contain only the committed answer(s), never
the rejected value. If the question is ambiguous because several distinct entities share the
same name, give the correct answer for each entity; that is not a conflict.

Return JSON only:
{
  "decision": "answered|unresolved_conflict|needs_information|insufficient_evidence",
  "answer": "Readable answer with all important conditions and exceptions; mention flagged dissent briefly.",
  "short_answers": ["each distinct final answer as a short span; for an unresolved conflict, each competing answer"],
  "citations": [
    {"chunk_id": "exact supplied chunk id", "quote": "verbatim excerpt from that passage",
     "treatment": "governing|supporting|superseded|different_scope|conflicting|uncertain",
     "explanation": "brief plain-language treatment"}
  ],
  "explanation": "Brief explanation of this decision, including how any conflict was resolved.",
  "missing_information": [],
  "clarification_question": null,
  "selected_source": null,
  "conflict_detected": false,
  "confidence": "high|medium|low"
}
For unresolved conflicts, cite each materially competing passage and explain its treatment;
selected_source must be null. For needs_information, list the missing facts and ask a useful
clarification question. Use an empty citation list if evidence cannot support any citation.
When the decision is answered, selected_source must be the filename of the governing passage the
answer is taken from (never null). selected_source must exactly match a supplied filename and
appear in citations. Numeric claim metadata is a search aid, never a citation or proof."""

# Plain evidence-grounded RAG prompt used by the ``baseline`` mode. Same output
# schema so results are directly comparable, but no governance guidance: the
# model is simply asked to answer from the retrieved passages.
BASELINE_SYSTEM_PROMPT = """You answer questions using only the supplied evidence passages and optional case context.
Read the passages and answer the question as directly as you can from them. If the passages
disagree, use your best judgment about which answer the evidence supports.

Return JSON only:
{
  "decision": "answered|unresolved_conflict|needs_information|insufficient_evidence",
  "answer": "Readable answer.",
  "short_answers": ["each distinct final answer as a short span"],
  "citations": [
    {"chunk_id": "exact supplied chunk id", "quote": "verbatim excerpt from that passage",
     "treatment": "supporting|conflicting",
     "explanation": "brief plain-language note"}
  ],
  "explanation": "Brief explanation of this answer.",
  "missing_information": [],
  "clarification_question": null,
  "selected_source": "filename of the passage the answer is taken from, or null",
  "conflict_detected": false,
  "confidence": "high|medium|low"
}
Use "answered" whenever the passages support an answer. When the decision is answered,
selected_source must be the filename of the passage the answer is taken from (never null); it must
exactly match a supplied filename and must also appear in citations."""

# Closed-book control (``closedbook`` mode): no retrieved passages at all. A
# targeted few-shot chain-of-thought prompt whose worked examples match the
# task format (single answer, several answers when a name is ambiguous, and an
# honest "unknown"), so the control is a fair test of parametric knowledge.
CLOSED_BOOK_SYSTEM_PROMPT = """You answer questions from your own knowledge. No documents are provided.
Think step by step first: identify what the question refers to, recall what you know, check whether the
name could refer to several distinct things, and decide how confident you are. Then give the answer.

Rules:
- If the question is ambiguous because several distinct entities share the same name, give the correct answer
  for each entity and list each one in short_answers with the entity in parentheses.
- If you do not know, say so: set decision to "insufficient_evidence" and leave short_answers empty.
  Do not guess a specific value you cannot recall.
- short_answers holds each distinct final answer as a short span (a name, a number, a date, a phrase).

Return the reasoning, then JSON only:
{
  "reasoning": "step-by-step thinking",
  "decision": "answered|insufficient_evidence",
  "answer": "Readable answer.",
  "short_answers": ["short span", "..."],
  "confidence": "high|medium|low"
}

Example 1
Question: Who wrote the novel Pride and Prejudice?
{
  "reasoning": "Pride and Prejudice is an English novel first published in 1813. Its author is Jane Austen. The title is not shared by another well-known work, so there is one answer.",
  "decision": "answered",
  "answer": "Pride and Prejudice was written by Jane Austen and published in 1813.",
  "short_answers": ["Jane Austen"],
  "confidence": "high"
}

Example 2
Question: What is the population of Georgia?
{
  "reasoning": "Georgia names two distinct places: the U.S. state and the country in the Caucasus. The question does not say which, so both answers are needed. The state had about 10.7 million residents in the 2020 census; the country has roughly 3.7 million people.",
  "decision": "answered",
  "answer": "The name is ambiguous. The U.S. state of Georgia had about 10.7 million residents in the 2020 census, while the country of Georgia has roughly 3.7 million people.",
  "short_answers": ["about 10.7 million (U.S. state of Georgia, 2020 census)", "about 3.7 million (country of Georgia)"],
  "confidence": "medium"
}

Example 3
Question: In what year was the Mill Creek covered bridge in Ashtabula County last repainted?
{
  "reasoning": "This asks for a specific maintenance date for a small local structure. I have no reliable knowledge of it and guessing a year would likely be wrong.",
  "decision": "insufficient_evidence",
  "answer": "I do not know when that bridge was last repainted.",
  "short_answers": [],
  "confidence": "low"
}"""

INITIAL_RETRIEVAL_K = 30
FINAL_CONTEXT_K = 8

# Reranking weights
W_SEMANTIC = 0.30
W_AUTHORITY = 0.50
W_SCOPE = 0.20

# Relevance gate. Authority and scope bonuses only apply to chunks whose
# semantic similarity is within this margin of the most relevant retrieved
# chunk, fading linearly to zero at the margin. Without it, an authoritative
# document about an unrelated topic (similarity 0.57 vs 0.82 for the relevant
# ones) outranked every relevant passage purely on its authority score and
# filled most of the context window. Set to None to disable (pure additive).
RELEVANCE_GATE_MARGIN = 0.10


def relevance_gate(semantic, best_semantic, margin=None):
    """Return a 0..1 multiplier for authority/scope bonuses."""
    margin = RELEVANCE_GATE_MARGIN if margin is None else margin
    if not margin or margin <= 0:
        return 1.0
    gate = (semantic - (best_semantic - margin)) / margin
    return max(0.0, min(1.0, gate))


# ============================================================
# DATA STRUCTURES (same as original)
# ============================================================

@dataclass
class Document:
    filename: str
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Candidate:
    filename: str
    text: str
    semantic_score: float
    authority_score: float
    scope_score: float
    claim: Optional[str]
    claim_sentence: Optional[str]
    status: str
    doc_type: str
    final_score: float = 0.0
    chunk_index: Optional[int] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    relevance_gate: float = 1.0


@dataclass
class Chunk:
    """A retrieval-unit slice of a document, preserving doc-level identity."""
    filename: str
    document_id: str
    chunk_index: int
    chunk_id: str
    text: str
    scope_text: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    scope_embedding: Optional[np.ndarray] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None


@dataclass
class DocumentMeta:
    """Governance metadata computed once per whole document, reused by every chunk."""
    status: str
    doc_type: str
    authority_raw: float


DECISIONS = {"answered", "unresolved_conflict", "needs_information", "insufficient_evidence"}
CITATION_TREATMENTS = {
    "governing", "supporting", "superseded", "different_scope", "conflicting", "uncertain",
}


def format_location(page_start=None, page_end=None, section=None):
    parts = []
    if page_start is not None:
        parts.append(f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}")
    if section:
        parts.append(f"section {section}")
    return ", ".join(parts) or "chunk reference only"


def generation_failure(reason):
    """A visible failure result. ``validation_error`` marks that this came from
    the pipeline rejecting the model's output, not from the model deciding the
    evidence was insufficient, so evaluations can tell the two apart."""
    return {
        "decision": "insufficient_evidence",
        "answer": "No answer could be safely generated from the retrieved evidence.",
        "citations": [],
        "reason": reason,
        "explanation": reason,
        "missing_information": [],
        "clarification_question": None,
        "selected_source": None,
        "conflict_detected": False,
        "confidence": "low",
        "validation_error": True,
    }


# Characters that models routinely drop or alter when quoting Markdown
# passages: emphasis markers, code ticks, heading/quote markers, and
# typographic quotes/dashes. Quote verification compares both sides after
# removing or unifying them, so a verbatim quote of "**7,430 Quens**" that
# comes back as "7,430 Quens" still verifies.
_QUOTE_STRIP = re.compile(r"[*_`#>\[\]]")
_QUOTE_UNIFY = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


def _normalized_quote(text):
    text = text or ""
    for src, dst in _QUOTE_UNIFY.items():
        text = text.replace(src, dst)
    text = _QUOTE_STRIP.sub("", text)
    text = " ".join(text.split()).casefold()
    # Models often wrap the quote in quotation marks or trailing ellipses.
    return text.strip(" \"'.")


def quote_matches(quote, text):
    """True if ``quote`` is a verbatim span of ``text`` after normalisation.

    Models often shorten a long passage by joining two or more non-adjacent
    spans with an ellipsis ("A ... B"). Each span is then verified separately
    and must appear in the passage in the same order.
    """
    q = _normalized_quote(quote)
    t = _normalized_quote(text)
    if not q:
        return False
    if q in t:
        return True
    segments = [seg.strip(" \"'.,;:") for seg in q.split("...")]
    segments = [seg for seg in segments if seg]
    if len(segments) < 2:
        return False
    pos = 0
    for seg in segments:
        idx = t.find(seg, pos)
        if idx < 0:
            return False
        pos = idx + len(seg)
    return True


def run_generation(complete, system_prompt, user_prompt, candidates, max_repairs=1):
    """Run a chat completion, validate it, and give the model one chance to
    repair a rejected output.

    ``complete(system_prompt, user_prompt) -> str`` is the provider call. On a
    validator rejection the rejection reason is appended to the prompt and the
    model is asked for corrected JSON. The returned result carries
    ``repair_attempts`` (0 or 1) and, for a final rejection, ``validation_error``.
    """
    feedback = ""
    for attempt in range(max_repairs + 1):
        response_text = complete(system_prompt, user_prompt + feedback)
        parsed = parse_json_response(response_text)
        if parsed is None:
            print("ERROR: JSON parsing failed")
            print(f"Full response ({len(response_text)} chars):")
            print(response_text)
            validated = generation_failure("Generation returned invalid JSON; no answer was selected.")
        else:
            validated = validate_generation_result(parsed, candidates)
        validated["repair_attempts"] = attempt
        if not validated.get("validation_error"):
            return validated
        if attempt == max_repairs:
            # Last resort: keep the answer but drop the citations that could
            # not be verified, rather than discarding a well-formed answer over
            # one bad quote. The number dropped is recorded for reporting.
            if parsed is not None:
                salvaged = validate_generation_result(parsed, candidates, drop_unverifiable=True)
                if not salvaged.get("validation_error"):
                    salvaged["repair_attempts"] = attempt
                    print(f"SALVAGED: dropped {salvaged.get('dropped_citations', 0)} unverifiable citation(s)")
                    return salvaged
            return validated
        print(f"REPAIR: asking the model to fix a rejected output ({validated['reason']})")
        feedback = (
            "\n\nYour previous JSON response was rejected for this reason: "
            f"{validated['reason']} "
            "Return the complete corrected JSON only. Every citation quote must be copied "
            "verbatim from the Content of the passage whose Chunk ID you cite, every "
            "treatment must be one of the allowed values, and selected_source must be a "
            "cited filename."
        )
    return validated


def format_evidence(candidates, include_signals=True):
    """Format candidates as numbered evidence blocks.

    ``include_signals=False`` omits the inferred status/type/authority/scope
    lines so a plain-RAG baseline sees only the passages themselves.
    """
    parts = []

    for i, c in enumerate(candidates, 1):
        chunk_label = f" (chunk {c.chunk_index + 1})" if c.chunk_index is not None else ""
        text = c.text if len(c.text) <= 6000 else c.text[:6000] + "..."
        signal_lines = ""
        if include_signals:
            signal_lines = (
                f"Inferred status: {c.status}\n"
                f"Inferred type: {c.doc_type}\n"
                f"Inferred authority signal: {c.authority_score:.3f}\n"
                f"Scope similarity (not proof of applicability): {c.scope_score:.3f}\n"
            )
        parts.append(f"""Evidence {i}{chunk_label}:
Chunk ID: {c.chunk_id}
Document ID: {c.document_id}
Filename: {c.filename}
{signal_lines}Location: {format_location(c.page_start, c.page_end, c.section)}

Content:
{text}

{'-'*80}
""")

    return "\n".join(parts)


def corroboration_summary(candidates):
    """Count, per extracted numeric claim, how many distinct documents carry it.

    A cheap, deterministic corroboration aid for the answer-and-flag policy:
    only claim-bearing (numeric) passages are counted, by filename, so the model
    can see e.g. "3,559: 2 documents; 10,000: 1 document". Returns "" when no
    claim appears in more than one competing form.
    """
    by_claim = defaultdict(set)
    for c in candidates:
        key = normalize_claim(c.claim)
        if key != "NO_CLAIM":
            by_claim[key].add(c.filename)
    if len(by_claim) < 2:
        return ""
    lines = sorted(((len(fs), claim) for claim, fs in by_claim.items()), reverse=True)
    return "Corroboration (numeric claims, distinct documents; count aid only):\n" + "\n".join(
        f"- {claim}: {n} document{'s' if n != 1 else ''}" for n, claim in lines
    )


def build_user_prompt(question, candidates, context=None, include_signals=True, corroboration=False):
    """Build the user turn sent to the generator (shared across providers)."""
    evidence_text = format_evidence(candidates, include_signals=include_signals)
    return f"""Question: {question}

Case context (user-provided, potentially incomplete):
{json.dumps(context or {}, ensure_ascii=True)}

Retrieved Evidence:

{evidence_text}
{(corroboration_summary(candidates) + chr(10) + chr(10)) if corroboration else ""}Analyze the evidence and provide your answer in JSON format."""


def parse_json_response(response_text):
    """Parse a JSON object out of a model response; returns None on failure.

    Tolerates markdown code fences and leading/trailing prose by falling back
    to the outermost ``{...}`` span.
    """
    text = (response_text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
        # Last resort: lenient repair (missing quotes/commas, truncated tail).
        # Anything it invents is still subject to citation validation.
        try:
            from json_repair import repair_json
            repaired = repair_json(text[start:], return_objects=True)
        except Exception:  # noqa: BLE001 - optional dependency / repair failure
            return None
        if isinstance(repaired, dict) and repaired.get("decision"):
            print("WARNING: model JSON was malformed and was repaired leniently")
            return repaired
        return None


def validate_generation_result(result, candidates, drop_unverifiable=False):
    """Reject generated citations and source names not grounded in retrieval.

    On rejection the raw model output is logged so that failures can be
    diagnosed from the run log instead of being silently coerced into
    ``insufficient_evidence``. With ``drop_unverifiable=True`` unverifiable
    citations are removed instead of rejecting the whole result; the count is
    returned as ``dropped_citations``.
    """
    validated = _validate_generation_result(result, candidates, drop_unverifiable=drop_unverifiable)
    if validated.get("validation_error"):
        try:
            raw = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = repr(result)
        print(f"VALIDATION REJECTED: {validated['reason']}")
        print(f"Raw model output ({len(raw)} chars): {raw[:4000]}")
    return validated


def _validate_generation_result(result, candidates, drop_unverifiable=False):
    if not isinstance(result, dict):
        return generation_failure("Generation returned an invalid structured result.")

    decision = result.get("decision")
    if decision not in DECISIONS:
        return generation_failure("Generation returned an unsupported decision.")

    candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
    # Models often cite the document id or filename instead of the full
    # "<doc>:chunk:<n>" id. Accept those when they resolve to exactly one
    # retrieved chunk; ambiguous or unknown ids are still rejected.
    by_alias = defaultdict(list)
    for candidate in candidates:
        for alias in {candidate.document_id, candidate.filename,
                      (candidate.filename or "").rsplit(".", 1)[0]}:
            if alias:
                by_alias[alias].append(candidate)
    for alias, matches in by_alias.items():
        if alias not in candidates_by_id and len(matches) == 1:
            candidates_by_id[alias] = matches[0]
    citations = result.get("citations", [])
    if not isinstance(citations, list):
        return generation_failure("Generation returned an invalid citation list.")

    verified_citations = []
    problems = []
    dropped = 0
    for position, citation in enumerate(citations, 1):
        if not isinstance(citation, dict):
            return generation_failure("Generation returned a malformed citation.")
        chunk_id = citation.get("chunk_id")
        candidate = candidates_by_id.get(chunk_id)
        quote = citation.get("quote")
        problem = None
        if candidate is None:
            problem = f"citation {position} cites chunk_id {chunk_id!r}, which is not among the retrieved passages"
        elif not isinstance(quote, str) or not quote.strip():
            problem = f"citation {position} ({chunk_id}) has an empty quote"
        elif not quote_matches(quote, candidate.text):
            snippet = " ".join(quote.split())
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            problem = (
                f"citation {position} ({chunk_id}) quote is not verbatim from that passage: \"{snippet}\""
            )
        if problem:
            if drop_unverifiable:
                dropped += 1
                continue
            problems.append(problem)
            continue
        treatment = citation.get("treatment", "supporting")
        if treatment not in CITATION_TREATMENTS:
            return generation_failure("Generation returned an unsupported evidence treatment.")
        verified_citations.append({
            "chunk_id": candidate.chunk_id,
            "document_id": candidate.document_id,
            "filename": candidate.filename,
            "page_start": candidate.page_start,
            "page_end": candidate.page_end,
            "section": candidate.section,
            "passage": candidate.text,
            "quote": quote,
            "treatment": treatment,
            "explanation": str(citation.get("explanation") or ""),
        })

    if problems:
        return generation_failure(
            "Generation cited evidence that was not retrieved or quoted text that does not match: "
            + "; ".join(problems)
            + "."
        )
    if decision in {"answered", "unresolved_conflict", "needs_information"} and not verified_citations:
        return generation_failure("Generation did not provide a verifiable evidence citation.")
    if decision == "unresolved_conflict" and not any(
        citation["treatment"] == "conflicting" for citation in verified_citations
    ):
        return generation_failure("Unresolved evidence was not identified as conflicting.")

    selected_source = result.get("selected_source")
    cited_filenames = {citation["filename"] for citation in verified_citations}
    if selected_source is not None and selected_source not in cited_filenames:
        if drop_unverifiable and dropped:
            selected_source = None
        else:
            return generation_failure("Generation selected a source that was not cited.")
    if decision == "answered" and selected_source is None:
        # Some models answer and cite a single governing (or sole supporting)
        # passage but leave selected_source null. Derive it from the citations
        # when, and only when, that choice is unambiguous.
        governing = {c["filename"] for c in verified_citations if c["treatment"] == "governing"}
        supporting = {c["filename"] for c in verified_citations if c["treatment"] == "supporting"}
        if len(governing) == 1:
            selected_source = next(iter(governing))
        elif not governing and len(supporting) == 1:
            selected_source = next(iter(supporting))
    if decision == "unresolved_conflict":
        selected_source = None

    missing = result.get("missing_information") or []
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        return generation_failure("Generation returned malformed missing-information details.")
    clarification = result.get("clarification_question")
    if clarification is not None and not isinstance(clarification, str):
        return generation_failure("Generation returned a malformed clarification question.")
    if decision == "needs_information" and (not missing or not clarification or not clarification.strip()):
        return generation_failure(
            "Generation identified missing case facts without listing them and asking a clarification question."
        )

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return generation_failure("Generation returned no readable answer.")
    if decision == "insufficient_evidence":
        answer = "The retrieved evidence does not support a reliable answer."
        selected_source = None

    explanation = result.get("explanation") or result.get("reason") or ""
    confidence = result.get("confidence", "unknown")
    if confidence not in {"high", "medium", "low"}:
        confidence = "unknown"
    short_answers = result.get("short_answers") or []
    if not isinstance(short_answers, list):
        short_answers = [short_answers]
    short_answers = [str(s).strip() for s in short_answers if str(s).strip()]
    if decision == "insufficient_evidence":
        short_answers = []

    return {
        "decision": decision,
        "answer": answer.strip(),
        "short_answers": short_answers,
        "citations": verified_citations,
        "reason": str(explanation),
        "explanation": str(explanation),
        "missing_information": missing,
        "clarification_question": clarification,
        "selected_source": selected_source,
        "conflict_detected": bool(result.get("conflict_detected"))
        or decision == "unresolved_conflict",
        "confidence": confidence,
        "dropped_citations": dropped,
    }


# ============================================================
# VERTEX AI / GEMINI EMBEDDINGS
# ============================================================

class GeminiEmbedder:
    def __init__(self, project_id=None, region=None, api_key=None):
        """
        Initialize embedder with Vertex AI (project-based) or AI Studio (API key).
        
        Priority:
        1. Vertex AI with project_id (recommended for GCP)
        2. AI Studio with api_key (fallback)
        """
        self.use_vertex = False
        
        # Try Vertex AI first (project-based authentication)
        if project_id or VERTEX_AI_PROJECT:
            try:
                project = project_id or VERTEX_AI_PROJECT
                location = region or VERTEX_AI_REGION
                
                vertexai.init(project=project, location=location)
                self.model = TextEmbeddingModel.from_pretrained(VERTEX_AI_MODEL)
                self.use_vertex = True
                
                print(f"✓ Using Vertex AI")
                print(f"  Project: {project}")
                print(f"  Region: {location}")
                print(f"  Model: {VERTEX_AI_MODEL}")
                print(f"  Auth: gcloud credentials")
                return
                
            except Exception as e:
                print(f"WARNING: Vertex AI initialization failed: {e}")
                print("Falling back to AI Studio API key authentication...")
        
        # Fallback to AI Studio (API key)
        if api_key or GEMINI_API_KEY:
            import google.generativeai as genai
            
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            print(f"✓ Using AI Studio API (API key)")
            self.use_vertex = False
        else:
            print("\nERROR: No authentication found!")
            print("\nOption 1 - Vertex AI (Recommended):")
            print("  gcloud auth application-default login")
            print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
            print("\nOption 2 - AI Studio:")
            print("  export GOOGLE_API_KEY='your-api-key'")
            print("  Get key from: https://makersuite.google.com/app/apikey")
            raise ValueError("No authentication configured")
    
    def encode(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        """
        Encode texts using Vertex AI or Gemini API.
        
        task_type:
            - RETRIEVAL_DOCUMENT: For indexing documents
            - RETRIEVAL_QUERY: For search queries
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.use_vertex:
            return self._encode_vertex(texts, task_type)
        else:
            return self._encode_genai(texts, task_type)
    
    def _encode_vertex(self, texts, task_type):
        """Encode using Vertex AI."""
        embeddings = []
        batch_size = 5  # Vertex AI has lower batch limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                # Vertex AI uses different method
                result = self.model.get_embeddings(batch)
                batch_embeddings = [emb.values for emb in result]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size}: {e}")
                dim = 768
                embeddings.extend([np.zeros(dim) for _ in batch])
        
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        
        return embeddings
    
    def _encode_genai(self, texts, task_type):
        """Encode using AI Studio API."""
        import google.generativeai as genai
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch,
                    task_type=task_type
                )
                
                batch_embeddings = result['embedding']
                
                if isinstance(batch_embeddings[0], (int, float)):
                    batch_embeddings = [batch_embeddings]
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error encoding batch {i//batch_size}: {e}")
                dim = 768
                embeddings.extend([np.zeros(dim) for _ in batch])
        
        embeddings = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        
        return embeddings


# ============================================================
# GEMINI GENERATOR (for final LLM-based resolution)
# ============================================================

class GeminiGenerator:
    """Generate final answers using Gemini 2.5 Pro."""
    
    def __init__(self, project_id=None, region=None, api_key=None):
        """Initialize Gemini generator with Vertex AI or AI Studio."""
        self.use_vertex = False
        
        # Try Vertex AI first
        if project_id or VERTEX_AI_PROJECT:
            try:
                from vertexai.generative_models import GenerativeModel
                
                project = project_id or VERTEX_AI_PROJECT
                location = region or VERTEX_AI_REGION
                
                vertexai.init(project=project, location=location)
                # Initialize model with system instruction
                self.model = GenerativeModel(
                    VERTEX_AI_GENERATION_MODEL,
                    system_instruction=[SYSTEM_PROMPT]
                )
                self.use_vertex = True
                
                print(f"✓ Using Vertex AI for generation")
                print(f"  Model: {VERTEX_AI_GENERATION_MODEL}")
                return
                
            except Exception as e:
                print(f"WARNING: Vertex AI generation init failed: {e}")
                print("Falling back to AI Studio...")
        
        # Fallback to AI Studio
        if api_key or GEMINI_API_KEY:
            import google.generativeai as genai
            
            key = api_key or GEMINI_API_KEY
            genai.configure(api_key=key)
            print(f"✓ Using AI Studio for generation")
            self.use_vertex = False
        else:
            raise ValueError("No authentication configured for generation")
    
    def _vertex_model(self, system_prompt):
        """Return a Vertex model bound to ``system_prompt`` (cached per prompt)."""
        if system_prompt == SYSTEM_PROMPT:
            return self.model
        cache = getattr(self, "_prompt_models", None)
        if cache is None:
            cache = self._prompt_models = {}
        if system_prompt not in cache:
            from vertexai.generative_models import GenerativeModel
            cache[system_prompt] = GenerativeModel(
                VERTEX_AI_GENERATION_MODEL, system_instruction=[system_prompt]
            )
        return cache[system_prompt]

    def generate(self, question, candidates, context=None, baseline=False):
        """
        Generate a structured decision and verify citations against retrieved chunks.

        Args:
            question: User's question
            candidates: List of Candidate objects (top K documents)
            context: optional case facts
            baseline: use the plain-RAG prompt without governance signals

        Returns:
            A validated decision object or a transparent insufficient-evidence result.
        """
        system_prompt = BASELINE_SYSTEM_PROMPT if baseline else SYSTEM_PROMPT
        user_prompt = build_user_prompt(
            question, candidates, context=context, include_signals=not baseline
        )

        try:
            return run_generation(self._complete, system_prompt, user_prompt, candidates)
        except Exception as e:
            print(f"ERROR: Gemini generation failed: {e}")
            return self._fallback_resolution(question, candidates)

    def _complete(self, system_prompt, user_prompt):
        """One chat completion; returns the raw response text."""
        if self.use_vertex:
            response = self._vertex_model(system_prompt).generate_content(
                [user_prompt],
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 4096,
                    "candidate_count": 1,
                }
            )
            try:
                return response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                return response.text

        # AI Studio
        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=system_prompt
        )
        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4096,
            )
        )
        return response.text
    
    def _format_evidence(self, candidates):
        """Kept for backward compatibility; see module-level ``format_evidence``."""
        return format_evidence(candidates, include_signals=True)

    def _fallback_resolution(self, question, candidates):
        """Return a visible failure rather than selecting a heuristic winner."""
        return generation_failure("Generation failed; no answer was selected.")


# ============================================================
# LOAD DOCUMENTS (same as original)
# ============================================================

def load_documents(folder: str) -> List[Document]:
    files = glob.glob(os.path.join(folder, "**/*.md"), recursive=True)
    docs = []
    
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        
        docs.append(Document(filename=os.path.basename(path), text=text))
    
    print(f"Loaded {len(docs)} documents")
    return docs


def build_document_meta(documents: List[Document]) -> Dict[str, DocumentMeta]:
    """Compute governance metadata once per whole document; chunks reuse it."""
    meta = {}
    for doc in documents:
        meta[doc.filename] = DocumentMeta(
            status=detect_status(doc.text),
            doc_type=detect_doc_type(doc.text),
            authority_raw=authority_score(doc.text),
        )
    return meta


def build_chunks(documents: List[Document]) -> List[Chunk]:
    """Split each document into retrieval-unit chunks (short docs -> 1 chunk)."""
    chunks = []
    for doc in documents:
        pieces = chunk_text_with_locations(doc.text)
        document_id = doc.filename.rsplit("__", 1)[-1]
        if document_id.lower().endswith(".md"):
            document_id = document_id[:-3]
        for idx, piece in enumerate(pieces):
            chunks.append(Chunk(
                filename=doc.filename,
                document_id=document_id,
                chunk_index=idx,
                chunk_id=f"{document_id}:chunk:{idx}",
                text=piece.text,
                page_start=piece.page_start,
                page_end=piece.page_end,
                section=piece.section,
            ))
    return chunks


# ============================================================
# UTILITY FUNCTIONS (same as original)
# ============================================================

def cosine_similarity(a, b):
    return float(np.dot(a, b))


def sentences(text):
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+|\n+", text)
        if x.strip()
    ]


# ============================================================
# UTILITY FUNCTIONS (Copied from gov_rag.py to avoid dependency)
# ============================================================

def sentences(text):
    """Split text into sentences"""
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+|\n+", text)
        if x.strip()
    ]


def keyword_overlap(question: str, sentence: str):
    """Calculate keyword overlap between question and sentence"""
    q_words = set(
        re.findall(r"[a-zA-Z]{3,}", question.lower())
    )
    
    s_words = set(
        re.findall(r"[a-zA-Z]{3,}", sentence.lower())
    )
    
    if not q_words:
        return 0
    
    return len(q_words & s_words) / len(q_words)


QUANTITY_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?:
        -?\d+(?:,\d{3})*(?:\.\d+)?
        \s*
        (?:
            %|
            °c|
            degrees?\s+celsius|
            minutes?|
            hours?|
            days?|
            years?|
            kilometers?|
            km|
            cases?|
            calls?|
            quens?|
            luma|
            inspection\s+points?
        )?
    )
    """,
    re.IGNORECASE | re.VERBOSE
)


# Header/first-N-characters window used to keep single ambiguous keywords
# (e.g. "draft", "summary") from firing on incidental body-text mentions in
# ordinary documents; strong multi-word phrases are still checked anywhere.
HEADER_WINDOW_CHARS = 1200


def get_header_window(text: str) -> str:
    return text[:HEADER_WINDOW_CHARS].lower()


def extract_structured_field(text: str, field_names) -> Optional[str]:
    """Match explicit 'Label: value' lines such as 'Status: DRAFT'."""
    pattern = re.compile(
        rf"^\s*(?:{'|'.join(field_names)})\s*:\s*(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).strip().lower() if match else None


def detect_status(text: str):
    """Detect document status conservatively: only trust an explicit structural
    cue (a 'Status:'/'Governing status:' line) or a strong, specific multi-word
    phrase, never a single ambiguous word (draft/archived/historical/stale)
    found anywhere in the body. Verified against all 180 ConflictBench Easy
    documents: every one of them is classified via a structured line or a
    strong phrase, so this conservative approach has zero regression there,
    while avoiding false positives on ordinary documents that happen to
    contain common words like "historical" or "propose" in their prose.
    """
    full = text.lower()
    structured = extract_structured_field(text, ["governing status", "status"])

    if "active and authoritative" in full or "governing status: active" in full:
        return "active_authoritative"
    if structured and "authoritative" in structured:
        return "active_authoritative"

    if any(x in full for x in ["superseded by", "withdrawn", "no longer valid"]):
        return "superseded"
    if structured and "supersed" in structured:
        return "superseded"

    if any(x in full for x in ["legacy version"]):
        return "archived"
    if structured and any(w in structured for w in ["archived", "historical", "legacy", "stale", "obsolete"]):
        return "archived"

    if any(x in full for x in ["draft policy", "working draft", "proposed policy", "proposed requirement", "planning brief"]):
        return "draft"
    if structured and any(w in structured for w in ["draft", "proposed"]):
        return "draft"

    if any(x in full for x in [
        "active governing policy",
        "governing policy",
        "currently effective",
        "current policy",
        "effective immediately",
        "active policy",
    ]):
        return "active"
    if structured == "active":
        return "active"

    return "unknown"


def detect_doc_type(text: str):
    """Detect document type. Strong, specific multi-word governance terms are
    checked anywhere in the document. Weak generic words (summary/digest/
    overview) only count near the top, since a research paper's body section
    titled "Results Summary" should not make the whole document a
    secondary_summary. Falls back to a light heuristic for ordinary
    documents rather than forcing an unrelated governance type.
    """
    full = text.lower()
    header = get_header_window(text)

    hierarchy = [
        ("governing_policy", ["governing policy", "official policy"]),
        ("policy_bulletin", ["policy bulletin"]),
        ("standard", ["official standard", "safety standard", "compliance standard"]),
        ("implementation_memo", ["implementation memo", "implementation memorandum"]),
        ("faq", ["faq", "frequently asked questions"]),
        ("training_guide", ["training guide", "training material"]),
        ("reference_card", ["reference card", "quick reference"]),
    ]

    for name, keywords in hierarchy:
        if any(k in full for k in keywords):
            return name

    if any(k in header for k in ["summary", "digest", "overview"]):
        return "secondary_summary"

    # Ordinary-document fallback: sensible generic types, not governance labels.
    if "abstract" in header and any(k in full for k in ["references", "et al", "arxiv"]):
        return "research_paper"
    if any(k in header for k in ["readme", "documentation", "user guide", "getting started"]):
        return "documentation"
    if any(k in full for k in ["technical report", "specification", "architecture overview"]):
        return "technical_document"

    return "unknown"


def authority_score(text: str):
    """Calculate authority score based on text markers"""
    t = text.lower()
    # Reuse detect_status's conservative structured/strong-phrase logic rather
    # than re-implementing bare-keyword checks here (those false-positive on
    # ordinary prose - see detect_status docstring).
    status_hint = detect_status(text)
    
    score = 0.0
    
    # Strongest signal: explicit active and authoritative marker
    if "active and authoritative" in t:
        score += 10.0
    
    if "governing status: active" in t:
        score += 8.0
    
    # Supersession language
    if "supersedes all previous" in t:
        score += 4.0
    elif "supersedes earlier" in t:
        score += 4.0
    elif "supersedes previous" in t:
        score += 3.0
    
    # Standard authority markers
    if "active governing policy" in t:
        score += 5.0
    elif "governing policy" in t:
        score += 4.0
    
    if "current policy" in t:
        score += 3.0
    
    if "currently effective" in t:
        score += 3.0
    
    if "effective immediately" in t:
        score += 2.5
    
    if "official policy" in t:
        score += 2.5
    
    if "policy bulletin" in t:
        score += 2.0
    
    if "official standard" in t:
        score += 2.0
    
    # ------------------------
    # Negative authority
    # ------------------------
    
    # Explicit disclaimers (very negative)
    if "should not be treated as the governing source" in t:
        score -= 8.0
    
    if "consult the active governing policy if a conflict exists" in t:
        score -= 6.0
    
    if "does not state the current requirement" in t:
        score -= 7.0
    
    # Status markers
    if "superseded" in t and "supersedes" not in t:
        score -= 6.0
    
    if status_hint == "archived":
        score -= 5.0
    
    if status_hint == "draft":
        score -= 4.0
    
    # Document type markers (secondary sources)
    if "secondary internal summary" in t:
        score -= 4.0
    
    if "implementation memo" in t:
        score -= 3.0
    
    if "training guide" in t:
        score -= 2.5
    
    if "planning brief" in t:
        score -= 2.5
    
    if "compliance digest" in t:
        score -= 2.0
    
    if "reference card" in t:
        score -= 2.0
    
    if "background note" in t:
        score -= 3.0
    
    if "faq" in t:
        score -= 1.5
    
    # Scope issues
    if "wrong scope" in t:
        score -= 5.0
    
    if "neighboring category" in t:
        score -= 4.0
    
    if "not the target category" in t:
        score -= 4.0
    
    return score


def extract_claim(question: str, text: str):
    """Extract numerical claim most related to question"""
    candidates = []
    
    for sent in sentences(text):
        matches = QUANTITY_PATTERN.findall(sent)
        
        if not matches:
            continue
        
        relevance = keyword_overlap(question, sent)
        
        for value in matches:
            value = value.strip()
            
            candidates.append(
                (
                    relevance,
                    value,
                    sent
                )
            )
    
    if not candidates:
        return None, None
    
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )
    
    _, value, sentence = candidates[0]
    
    return value, sentence


def normalize_claim(claim):
    """Normalize claim text for comparison"""
    if not claim:
        return "NO_CLAIM"
    
    x = claim.lower().strip()
    
    x = x.replace(",", "")
    x = x.replace("kilometers", "km")
    x = x.replace("kilometer", "km")
    x = x.replace("degrees celsius", "°c")
    
    x = re.sub(r"\s+", " ", x)
    
    return x


def normalize_authority(raw):
    """Normalize authority score to 0-1 range"""
    # expected approximate range [-10, +15]
    raw = max(-10, min(15, raw))
    
    return (raw + 10) / 25


def group_by_claim(candidates):
    """Group candidates by normalized claim"""
    groups = defaultdict(list)
    
    for candidate in candidates:
        key = normalize_claim(candidate.claim)
        groups[key].append(candidate)
    
    return groups


# ============================================================
# RETRIEVAL (adapted for Gemini)
# ============================================================

def semantic_retrieve(question, chunks, embedder, k=INITIAL_RETRIEVAL_K):
    """Broad semantic retrieval over chunks using Gemini embeddings."""
    
    # Encode query with RETRIEVAL_QUERY task type
    q_embedding = embedder.encode([question], task_type="RETRIEVAL_QUERY")[0]
    
    scored = []
    
    for chunk in chunks:
        score = cosine_similarity(q_embedding, chunk.embedding)
        scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k], q_embedding


_BM25_TOKEN = re.compile(r"\w+")


def hybrid_retrieve(question, chunks, embedder, k=INITIAL_RETRIEVAL_K, rrf_k=60):
    """Hybrid retrieval control: BM25 and dense rankings fused with reciprocal
    rank fusion (``1/(rrf_k + rank)``, summed over the two rankers).

    Returns ``(retrieved, q_embedding, fused)`` where ``retrieved`` is the
    fused top-k as ``(cosine_score, chunk)`` pairs so downstream code sees the
    same shape as ``semantic_retrieve``, and ``fused`` maps chunk_id to its
    fusion score for use as the final ranking score.
    """
    from rank_bm25 import BM25Okapi

    q_embedding = embedder.encode([question], task_type="RETRIEVAL_QUERY")[0]
    dense = sorted(
        ((cosine_similarity(q_embedding, c.embedding), c) for c in chunks),
        key=lambda x: x[0], reverse=True,
    )
    cosine_by_id = {c.chunk_id: score for score, c in dense}

    tokenised = [_BM25_TOKEN.findall(c.text.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenised)
    bm25_scores = bm25.get_scores(_BM25_TOKEN.findall(question.lower()))
    lexical = sorted(zip(bm25_scores, chunks), key=lambda x: x[0], reverse=True)

    fused = {}
    for ranking in (dense, lexical):
        for rank, (_, chunk) in enumerate(ranking, start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    by_id = {c.chunk_id: c for c in chunks}
    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    retrieved = [(cosine_by_id[cid], by_id[cid]) for cid, _ in top]
    return retrieved, q_embedding, fused


def first_8_sentence_scope_text(document_text):
    """Return the scope text used for reranking: first 8 sentences of a chunk."""
    return " ".join(sentences(document_text)[:8])


def get_scope_score(chunk, q_embedding):
    """Score scope match using the chunk's persisted scope embedding."""
    if chunk.scope_embedding is None:
        return 0.0
    return cosine_similarity(q_embedding, chunk.scope_embedding)


# ============================================================
# CANDIDATE BUILDING (same logic, different embedder)
# ============================================================

def build_candidates(question, retrieved, q_embedding, doc_meta):
    """Build ranking candidates from retrieved chunks.

    Governance metadata (status/doc_type/authority) is looked up per
    document_id and reused across every chunk of that document; when a
    document has no explicit governance signals, doc_meta yields neutral
    defaults (status=unknown, authority=0.0) so semantic/scope scores
    dominate ranking naturally rather than being penalized.
    """
    candidates = []
    default_meta = DocumentMeta(status="unknown", doc_type="unknown", authority_raw=0.0)
    best_semantic = max((semantic for semantic, _ in retrieved), default=0.0)

    for semantic, chunk in retrieved:
        meta = doc_meta.get(chunk.filename, default_meta)
        auth = normalize_authority(meta.authority_raw)
        scope = get_scope_score(chunk, q_embedding)
        claim, claim_sentence = extract_claim(question, chunk.text)
        gate = relevance_gate(semantic, best_semantic)

        candidate = Candidate(
            filename=chunk.filename,
            text=chunk.text,
            semantic_score=semantic,
            authority_score=auth,
            scope_score=scope,
            claim=claim,
            claim_sentence=claim_sentence,
            status=meta.status,
            doc_type=meta.doc_type,
            chunk_index=chunk.chunk_index,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section=chunk.section,
            relevance_gate=gate,
        )

        # Authority and scope can only reorder chunks that are comparably
        # relevant; they cannot lift an off-topic chunk above on-topic ones.
        candidate.final_score = (
            W_SEMANTIC * candidate.semantic_score +
            gate * (
                W_AUTHORITY * candidate.authority_score +
                W_SCOPE * candidate.scope_score
            )
        )
        
        candidates.append(candidate)
    
    return candidates


def diversify_conflicts(candidates, max_per_claim=1, max_no_claim=None):
    """Reduce duplicate claim passages without collapsing distinct conditions.

    Claim-bearing candidates are grouped by both the extracted number and full
    normalized passage, so exceptions remain independently reviewable. Chunks
    without a numeric claim are retained up to their separate context limit.
    """
    groups = defaultdict(list)
    for candidate in candidates:
        claim_key = normalize_claim(candidate.claim)
        if claim_key == "NO_CLAIM":
            groups[claim_key].append(candidate)
            continue
        # A repeated number can appear with a materially different condition
        # or exception; keep those as independent evidence groups.
        passage_key = " ".join(candidate.text.lower().split())
        groups[(claim_key, passage_key)].append(candidate)
    diversified = []
    
    for claim, docs in groups.items():
        docs.sort(key=lambda x: x.final_score, reverse=True)
        if claim == "NO_CLAIM":
            limit = max_no_claim if max_no_claim is not None else len(docs)
        else:
            limit = max_per_claim
        diversified.extend(docs[:limit])
    
    diversified.sort(key=lambda x: x.final_score, reverse=True)
    return diversified


def authority_resolve(candidates):
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: c.final_score, reverse=True)[0]


def print_conflict_table(candidates):
    """Display candidates in table format."""
    print("\n" + "=" * 100)
    print(
        f"{'FILE':30} {'CLAIM':15} {'STATUS':12} {'TYPE':20} "
        f"{'SEM':7} {'AUTH':7} {'SCOPE':7} {'FINAL':7}"
    )
    print("=" * 100)
    
    for c in candidates:
        print(
            f"{c.filename[:30]:30} "
            f"{str(c.claim)[:15]:15} "
            f"{c.status[:12]:12} "
            f"{c.doc_type[:20]:20} "
            f"{c.semantic_score:.3f}   "
            f"{c.authority_score:.3f}   "
            f"{c.scope_score:.3f}   "
            f"{c.final_score:.3f}"
        )


# ============================================================
# GOV-RAG with Gemini
# ============================================================

class GovRAGGemini:
    def __init__(
        self,
        doc_dir,
        project_id=None,
        region=None,
        api_key=None,
        use_llm=True,
        embedder=None,
        generator=None,
        documents=None,
    ):
        """Initialize GOV-RAG with Gemini embeddings and optional LLM generation.

        Args:
            doc_dir: Path to document directory
            project_id: GCP project ID for Vertex AI
            region: GCP region (default: us-central1)
            api_key: API key for AI Studio (fallback)
            use_llm: If True, use an LLM for final answer generation (default: True)
            embedder: optional drop-in embedder (e.g. ``providers.LocalEmbedder``);
                defaults to ``GeminiEmbedder``
            generator: optional drop-in generator (e.g.
                ``providers.OpenAICompatGenerator``); defaults to ``GeminiGenerator``
        """

        print("\nInitializing GOV-RAG...")

        self.use_llm = use_llm
        self.embedder = embedder or GeminiEmbedder(project_id=project_id, region=region, api_key=api_key)

        # Initialize generator if using LLM
        if not self.use_llm:
            self.generator = None
        elif generator is not None:
            self.generator = generator
        else:
            print("\nInitializing Gemini generator for final resolution...")
            self.generator = GeminiGenerator(project_id=project_id, region=region, api_key=api_key)
        
        # Documents may be supplied in memory (``documents=[Document(...)]``),
        # e.g. for per-question corpora from external benchmarks; otherwise
        # they are loaded from ``doc_dir``.
        self.documents = list(documents) if documents is not None else load_documents(doc_dir)
        self.doc_meta = build_document_meta(self.documents)
        self.chunks = build_chunks(self.documents)
        
        # Create embeddings for all chunks (short docs collapse to one chunk
        # each, so this is equivalent to prior whole-document behavior there).
        print(f"Creating embeddings for {len(self.chunks)} chunks across {len(self.documents)} documents...")
        texts = [c.text for c in self.chunks]
        
        embeddings = self.embedder.encode(texts, task_type="RETRIEVAL_DOCUMENT")
        
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding

        # Scope embeddings are used during reranking. Persisting them keeps
        # query-time retrieval to one query embedding plus local vector math.
        print(f"Creating scope embeddings for {len(self.chunks)} chunks...")
        scope_texts = [first_8_sentence_scope_text(c.text) for c in self.chunks]
        non_empty_scope_items = [(idx, text) for idx, text in enumerate(scope_texts) if text]
        scope_embeddings_by_index = {}

        if non_empty_scope_items:
            scope_embeddings = self.embedder.encode(
                [text for _, text in non_empty_scope_items],
                task_type="RETRIEVAL_DOCUMENT",
            )
            scope_embeddings_by_index = {
                idx: embedding for (idx, _), embedding in zip(non_empty_scope_items, scope_embeddings)
            }

        for idx, chunk in enumerate(self.chunks):
            chunk.scope_text = scope_texts[idx]
            chunk.scope_embedding = scope_embeddings_by_index.get(
                idx,
                np.zeros_like(chunk.embedding),
            )
        
        print("Ready.")

    def to_persisted_index_rows(self) -> List[Dict]:
        """Serialize chunk embeddings plus document governance metadata."""
        rows = []
        for chunk in self.chunks:
            meta = self.doc_meta.get(
                chunk.filename,
                DocumentMeta(status="unknown", doc_type="unknown", authority_raw=0.0),
            )
            embedding = chunk.embedding
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            scope_embedding = chunk.scope_embedding
            if hasattr(scope_embedding, "tolist"):
                scope_embedding = scope_embedding.tolist()
            scope_text = chunk.scope_text if chunk.scope_text is not None else first_8_sentence_scope_text(chunk.text)

            rows.append({
                "chunk_id": chunk.chunk_id,
                "chunk_index": int(chunk.chunk_index),
                "filename": chunk.filename,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "scope_text": scope_text,
                "status": meta.status,
                "doc_type": meta.doc_type,
                "authority_raw": float(meta.authority_raw),
                "embedding": list(embedding or []),
                "scope_embedding": list(scope_embedding or []),
            })
        return rows

    @classmethod
    def from_persisted_index(
        cls,
        rows: List[Dict],
        project_id=None,
        region=None,
        api_key=None,
        use_llm=True,
        embedder=None,
        generator=None,
    ):
        """Create a query-ready GOV-RAG instance from stored chunk embeddings.

        ``embedder`` / ``generator`` are drop-in providers, exactly as in
        ``__init__``; without them the Gemini classes are used.
        """
        print("\nInitializing GOV-RAG from persisted index...")

        obj = cls.__new__(cls)
        obj.use_llm = use_llm
        obj.embedder = embedder or GeminiEmbedder(project_id=project_id, region=region, api_key=api_key)

        if not obj.use_llm:
            obj.generator = None
        elif generator is not None:
            obj.generator = generator
        else:
            print("\nInitializing Gemini generator for final resolution...")
            obj.generator = GeminiGenerator(project_id=project_id, region=region, api_key=api_key)

        obj.documents = []
        obj.doc_meta = {}
        obj.chunks = []

        for row in rows:
            embedding = row.get("embedding") or []
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            scope_embedding = row.get("scope_embedding")
            if scope_embedding is None:
                raise ValueError("Persisted index row missing scope_embedding")
            if hasattr(scope_embedding, "tolist"):
                scope_embedding = scope_embedding.tolist()

            filename = row["filename"]
            obj.chunks.append(
                Chunk(
                    filename=filename,
                    document_id=row.get("document_id") or filename,
                    chunk_index=int(row.get("chunk_index", 0)),
                    chunk_id=row.get("chunk_id") or f"{filename}::chunk-{row.get('chunk_index', 0)}",
                    text=row.get("text") or row.get("scope_text") or "",
                    scope_text=row.get("scope_text"),
                    embedding=np.asarray(embedding, dtype=float),
                    scope_embedding=np.asarray(scope_embedding, dtype=float),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    section=row.get("section"),
                )
            )
            obj.doc_meta[filename] = DocumentMeta(
                status=row.get("status") or "unknown",
                doc_type=row.get("doc_type") or "unknown",
                authority_raw=float(row.get("authority_raw") or 0.0),
            )

        print(f"Ready from persisted index with {len(obj.chunks)} chunks.")
        return obj
    
    def query(self, question, context=None, mode="govrag", retrieval="hybrid", policy="answer_and_flag"):
        """Query the unified GOV-RAG pipeline (chunk retrieval + governance-aware
        reranking + conflict-aware Gemini reasoning). The same pipeline runs
        every time; when evidence agrees or has no governance metadata, it
        naturally reduces to ordinary evidence-grounded semantic RAG.

        ``mode="baseline"`` runs a plain-RAG control for evaluation: the same
        semantic retrieval, but the top-K chunks are taken by semantic score
        alone (no authority/scope reranking, no claim diversification) and the
        generator is given only the passages, with no governance guidance.

        Pipeline:
        1. Question
        2. Broad semantic retrieval over chunks (top ~30)
        3. Claim extraction (auxiliary metadata only, not gating)
        4. Governance metadata reuse (per-document, applied to its chunks)
        5. Authority + scope scoring
        6. Conflict-crowding control (only for genuine claim-bearing chunks)
        7. Top evidence chunks
        8. GEMINI 2.5 PRO (if use_llm=True)
        9. Final answer + supporting/governing source
        """
        
        # 1. Broad semantic retrieval over chunks
        context = {
            key: value for key, value in (context or {}).items()
            if value is not None and str(value).strip()
        }
        context_text = " ".join(f"{key}: {value}" for key, value in context.items())
        retrieval_query = f"{question}\nCase context: {context_text}" if context_text else question

        if mode == "closedbook":
            # Direct few-shot CoT control: no retrieval, no passages.
            if not (self.use_llm and self.generator and hasattr(self.generator, "closed_book")):
                failed = generation_failure("Closed-book mode needs a generator with closed_book().")
                return {**failed, "source": None, "method": "none",
                        "retrieval_mode": "closed_book", "candidates": []}
            print("\n" + "=" * 80 + "\nFINAL RESOLUTION: closed-book few-shot CoT (no retrieval)\n" + "=" * 80)
            llm_result = self.generator.closed_book(question, context=context)
            return {
                "answer": llm_result.get("answer"),
                "source": None,
                "decision": llm_result.get("decision", "insufficient_evidence"),
                "citations": [],
                "missing_information": [],
                "clarification_question": None,
                "explanation": llm_result.get("explanation", ""),
                "reason": llm_result.get("reason"),
                "conflict_detected": False,
                "confidence": llm_result.get("confidence", "medium"),
                "short_answers": llm_result.get("short_answers", []),
                "validation_error": bool(llm_result.get("validation_error", False)),
                "repair_attempts": int(llm_result.get("repair_attempts", 0)),
                "dropped_citations": 0,
                "method": "llm",
                "retrieval_mode": "closed_book",
                "candidates": [],
            }

        fused_scores = None
        # ``retrieval="hybrid"`` gives GOV-RAG the same BM25+dense RRF first stage
        # as the hybrid control; governance reranking still runs on top of it.
        if mode == "hybrid" or (mode == "govrag" and retrieval == "hybrid"):
            retrieved, q_embedding, fused_scores = hybrid_retrieve(
                retrieval_query, self.chunks, self.embedder, INITIAL_RETRIEVAL_K
            )
        else:
            retrieved, q_embedding = semantic_retrieve(
                retrieval_query, self.chunks, self.embedder, INITIAL_RETRIEVAL_K
            )
        
        # 2. Authority + scope extraction
        candidates = build_candidates(question, retrieved, q_embedding, self.doc_meta)

        baseline = mode in ("baseline", "hybrid")
        retrieval_mode = {"baseline": "baseline_semantic", "hybrid": "baseline_hybrid_rrf"}.get(mode, "unified_govrag")
        if mode == "govrag":
            retrieval_mode = f"unified_govrag_{retrieval}_{policy}"
        if baseline:
            # Plain RAG controls: rank by retrieval score only (cosine, or the
            # BM25+dense fusion score in hybrid mode), keep top-K.
            for candidate in candidates:
                if fused_scores is not None:
                    candidate.final_score = fused_scores.get(candidate.chunk_id, 0.0)
                else:
                    candidate.final_score = candidate.semantic_score
            diversified = sorted(candidates, key=lambda c: c.final_score, reverse=True)
            diversified = diversified[:FINAL_CONTEXT_K]
        else:
            # 3. Conflict diversification (claim-bearing chunks capped per claim;
            #    ordinary no-claim evidence is not collapsed to a single chunk).
            diversified = diversify_conflicts(candidates, max_per_claim=1, max_no_claim=FINAL_CONTEXT_K)
            diversified = diversified[:FINAL_CONTEXT_K]

        print_conflict_table(diversified)

        if not diversified:
            result = generation_failure("No relevant evidence was retrieved.")
            return {
                **result,
                "source": None,
                "method": "none",
                "retrieval_mode": retrieval_mode,
                "candidates": [],
            }

        # 4. Final resolution: LLM or rule-based
        if self.use_llm and self.generator:
            print("\n" + "="*80)
            print(f"FINAL RESOLUTION: {'plain RAG baseline' if baseline else 'GOV-RAG'} generation")
            print("="*80)

            gen_kwargs = {"policy": policy} if (not baseline and policy != "strict") else {}
            llm_result = self.generator.generate(
                question, diversified, context=context, baseline=baseline, **gen_kwargs
            )

            return {
                "answer": llm_result.get("answer"),
                "source": llm_result.get("selected_source"),
                "decision": llm_result.get("decision", "insufficient_evidence"),
                "citations": llm_result.get("citations", []),
                "missing_information": llm_result.get("missing_information", []),
                "clarification_question": llm_result.get("clarification_question"),
                "explanation": llm_result.get("explanation", ""),
                "reason": llm_result.get("reason"),
                "conflict_detected": llm_result.get("conflict_detected", False),
                "confidence": llm_result.get("confidence", "medium"),
                "short_answers": llm_result.get("short_answers", []),
                "validation_error": bool(llm_result.get("validation_error", False)),
                "repair_attempts": int(llm_result.get("repair_attempts", 0)),
                "dropped_citations": int(llm_result.get("dropped_citations", 0) or 0),
                "method": "llm",
                "retrieval_mode": retrieval_mode,
                "candidates": [asdict(x) for x in diversified]
            }
        else:
            failed = generation_failure(
                "Generation is disabled; retrieved evidence is shown without selecting an answer."
            )
            return {
                **failed,
                "source": None,
                "method": "generation_disabled",
                "retrieval_mode": retrieval_mode,
                "candidates": [asdict(x) for x in diversified]
            }


# ============================================================
# MAIN - Interactive Mode
# ============================================================

def interactive_mode(doc_dir, project_id=None, region=None, api_key=None, use_llm=True):
    """Run GOV-RAG in interactive mode with Vertex AI or Gemini."""
    rag = GovRAGGemini(doc_dir, project_id=project_id, region=region, api_key=api_key, use_llm=use_llm)
    
    while True:
        question = input("\nQuestion (or 'exit'): ").strip()
        
        if question.lower() == "exit":
            break
        
        result = rag.query(question)
        
        print("\nGOV-RAG RESULT (Gemini)")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        doc_dir = sys.argv[1]
        project_id = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        print("Usage: python pipeline/gov_rag_gemini.py <document_directory> [project_id]")
        print("\nExample with Vertex AI:")
        print("  export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("  gcloud auth application-default login")
        print("  python pipeline/gov_rag_gemini.py conflictbench_fictional_full/packs/easy")
        print("\nOr with AI Studio:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        print("  python pipeline/gov_rag_gemini.py conflictbench_fictional_full/packs/easy")
        sys.exit(1)
    
    interactive_mode(doc_dir, project_id=project_id)
