"""Locate verified quotes and chunks inside their source text.

The generator's citations are verified against retrieved chunks with the same
normalisation used here (casefold, unified quotation marks and dashes,
collapsed whitespace), so a quote that passed validation can always be mapped
back to character offsets in the chunk, and almost always in the full document.
The UI only marks ranges; it never re-searches text itself.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_UNIFY = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}
_ELLIPSIS = re.compile(r"…|\.\.\.")


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """Return (normalized_text, offsets) where offsets[i] is the index in
    ``text`` of the character that produced normalized character i."""
    out: List[str] = []
    offsets: List[int] = []
    pending_space = False
    for index, char in enumerate(text):
        char = _UNIFY.get(char, char)
        if char.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            offsets.append(index)
            pending_space = False
        for folded in char.casefold():
            out.append(folded)
            offsets.append(index)
    return "".join(out), offsets


def _normalize(text: str) -> str:
    return _normalize_with_map(text)[0]


def _segments(quote: str) -> List[str]:
    parts = [_normalize(part).strip(" \"'.,;:") for part in _ELLIPSIS.split(quote or "")]
    return [part for part in parts if part]


def locate_quote(quote: str, text: str, hint: Optional[int] = None) -> List[Tuple[int, int]]:
    """Character spans of ``quote`` in ``text`` (one span per ellipsis segment).

    Returns an empty list when the quote cannot be found. With several
    occurrences, ``hint`` (an approximate offset in ``text``) picks the nearest.
    """
    if not quote or not text:
        return []
    normalized, offsets = _normalize_with_map(text)
    spans: List[Tuple[int, int]] = []
    position = 0
    for segment in _segments(quote):
        starts = [m.start() for m in re.finditer(re.escape(segment), normalized)]
        starts = [s for s in starts if s >= position] or starts
        if not starts:
            return []
        if hint is not None and len(starts) > 1:
            start = min(starts, key=lambda s: abs(offsets[s] - hint))
        else:
            start = starts[0]
        end = start + len(segment)
        spans.append((offsets[start], offsets[end - 1] + 1))
        position = end
    return spans


def locate_chunk(chunk_text: str, document_text: str, hint: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Approximate span of a chunk in its document.

    Chunks are rebuilt from paragraphs and sentences joined with single spaces,
    so they are not exact substrings; the first and last few words are located
    instead, which is enough to draw boundaries.
    """
    words = (chunk_text or "").split()
    if not words:
        return None
    head = " ".join(words[:12])
    tail = " ".join(words[-12:])
    head_spans = locate_quote(head, document_text, hint=hint)
    if not head_spans:
        return None
    start = head_spans[0][0]
    tail_spans = locate_quote(tail, document_text, hint=start + len(chunk_text))
    tail_spans = [s for s in tail_spans if s[1] > start] or tail_spans
    end = tail_spans[0][1] if tail_spans else min(len(document_text), start + len(chunk_text))
    return (start, max(end, start + 1))


def locate_chunks(chunks: List[Dict], document_text: str) -> List[Dict]:
    """Attach ``start``/``end`` document offsets to chunk dicts (in order)."""
    located: List[Dict] = []
    cursor = 0
    for chunk in sorted(chunks, key=lambda c: int(c.get("chunk_index", 0))):
        span = locate_chunk(chunk.get("text", ""), document_text, hint=cursor)
        entry = dict(chunk)
        if span:
            entry["start"], entry["end"] = span
            cursor = span[0] + max(1, (span[1] - span[0]) // 2)
        else:
            entry["start"] = entry["end"] = None
        located.append(entry)
    return located
