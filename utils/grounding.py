"""Lightweight keyword overlap: note chunks, top excerpts, and disclaimer heuristics."""

from __future__ import annotations

import re
from typing import Final

_MIN_WORD_LEN: Final[int] = 3
_MAX_CHUNK_CHARS: Final[int] = 900
_MAX_EXCERPT_CHARS: Final[int] = 320
_DEFAULT_TOP_N: Final[int] = 3

DISCLAIMER_STANDARD = (
    "This response is generated from the provided notes. "
    "If the notes are incomplete, the answer may be partial."
)
DISCLAIMER_STRONG = (
    "The provided notes may not fully cover this question, "
    "so this answer could be incomplete."
)


def _tokens(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
        if len(w) >= _MIN_WORD_LEN
    }


def split_note_chunks(notes: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split notes on blank lines, then subdivide oversized blocks."""
    raw = [p.strip() for p in re.split(r"\n\s*\n", notes) if p.strip()]
    chunks: list[str] = []
    for para in raw:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sent in sentences:
            if not sent:
                continue
            if len(buf) + len(sent) + 1 <= max_chars:
                buf = f"{buf} {sent}".strip() if buf else sent.strip()
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = sent.strip() if len(sent) <= max_chars else sent.strip()[:max_chars]
        if buf:
            chunks.append(buf.strip())
    if not chunks and notes.strip():
        text = notes.strip()
        start = 0
        while start < len(text):
            chunks.append(text[start : start + max_chars])
            start += max_chars
    return chunks


def keyword_overlap_score(question_tokens: set[str], chunk: str) -> int:
    if not question_tokens:
        return 0
    return len(question_tokens & _tokens(chunk))


def top_keyword_excerpts(
    notes: str,
    question: str,
    *,
    n: int = _DEFAULT_TOP_N,
    max_excerpt_chars: int = _MAX_EXCERPT_CHARS,
) -> list[str]:
    """Return up to n short excerpts from notes with highest token overlap with the question."""
    qt = _tokens(question)
    chunks = split_note_chunks(notes)
    if not chunks:
        return []

    scored = sorted(
        ((keyword_overlap_score(qt, c), c) for c in chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    out: list[str] = []
    seen_norm: set[str] = set()
    for score, chunk in scored:
        norm = chunk[:200]
        if norm in seen_norm:
            continue
        if score == 0:
            break
        seen_norm.add(norm)
        snippet = chunk[:max_excerpt_chars] + ("…" if len(chunk) > max_excerpt_chars else "")
        out.append(snippet)
        if len(out) >= n:
            break

    # If nothing matched (e.g. very short question tokens), show the start of the notes.
    if not out:
        for c in chunks[:n]:
            snippet = c[:max_excerpt_chars] + ("…" if len(c) > max_excerpt_chars else "")
            out.append(snippet)
    return out


def weak_question_note_match(notes: str, question: str) -> bool:
    """Heuristic: low overlap between question keywords and best-matching chunk."""
    qt = _tokens(question)
    chunks = split_note_chunks(notes)
    if not qt or not chunks:
        return True
    max_s = max(keyword_overlap_score(qt, c) for c in chunks)
    ratio = max_s / max(len(qt), 1)
    if max_s <= 1 and len(qt) >= 4:
        return True
    if ratio < 0.22 and len(qt) >= 3:
        return True
    return False


def disclaimer_text(weak_match: bool) -> str:
    return DISCLAIMER_STRONG if weak_match else DISCLAIMER_STANDARD
