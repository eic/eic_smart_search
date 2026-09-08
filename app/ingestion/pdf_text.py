"""Shared PDF → text extraction for connectors.

Prefers PyMuPDF (``fitz``) — much better text ordering on the two-column
physics layouts common in Zenodo/Indico material, and ~10x faster than
pypdf. Falls back to pypdf when PyMuPDF is unavailable so the stack still
works on minimal installs.

Returns ``(text, meta)`` where meta records how extraction went; callers
store it under the document metadata for debuggability.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# PostgreSQL text columns reject NUL bytes, and PDFs routinely leak them
# along with other C0 control chars.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_text(text: str) -> str:
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub("", text)


def extract_pdf_text(data: bytes, *, max_chars: int) -> tuple[str, dict[str, Any]]:
    """Extract text from PDF bytes, truncated to ``max_chars``.

    pypdf is a true fallback: it runs when PyMuPDF is missing OR failed to
    parse the file — mildly corrupt PDFs one engine rejects are often
    readable by the other.
    """
    text, meta = _extract_with_pymupdf(data, max_chars=max_chars)
    if not text and meta.get("skipped"):
        text, meta = _extract_with_pypdf(data, max_chars=max_chars)
    return text, meta


def extract_pptx_text(data: bytes, *, max_chars: int) -> tuple[str, dict[str, Any]]:
    """Extract text from PowerPoint bytes (slides, tables, speaker notes).

    Indico decks are frequently uploaded as .pptx rather than PDF — the
    collaboration status presentations especially. One line per shape,
    slides separated by blank lines, notes prefixed so they stay
    attributable.
    """
    import io

    try:
        from pptx import Presentation  # type: ignore[import-not-found]
    except ImportError:
        return "", {"skipped": "python-pptx-missing"}
    try:
        prs = Presentation(io.BytesIO(data))
    except Exception as exc:
        return "", {"engine": "python-pptx", "skipped": "pptx-parse-failed", "error": str(exc)[:200]}

    parts: list[str] = []
    chars = 0
    slides_read = 0
    for slide in prs.slides:
        if chars >= max_chars:
            break
        lines: list[str] = []
        for shape in slide.shapes:
            try:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = "".join(run.text for run in para.runs).strip()
                        if text:
                            lines.append(text)
                elif getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            lines.append(" | ".join(cells))
            except Exception:
                continue
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    lines.append(f"[Notes] {notes}")
        except Exception:
            pass
        if lines:
            slides_read += 1
            block = "\n".join(lines)
            parts.append(block[: max(0, max_chars - chars)])
            chars += len(block)
    full = sanitize_text("\n\n".join(parts))
    return full, {
        "engine": "python-pptx",
        "total_slides": len(prs.slides._sldIdLst) if hasattr(prs.slides, "_sldIdLst") else slides_read,
        "slides_extracted": slides_read,
        "chars_extracted": len(full),
        "truncated": chars >= max_chars,
    }


def _extract_with_pymupdf(data: bytes, *, max_chars: int) -> tuple[str, dict[str, Any]]:
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
    except ImportError:
        return "", {"skipped": "pymupdf-missing"}
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        return "", {"engine": "pymupdf", "skipped": "pdf-parse-failed", "error": str(exc)[:200]}
    chunks: list[str] = []
    char_budget = max_chars
    pages_read = 0
    try:
        for page in doc:
            if char_budget <= 0:
                break
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            if not text.strip():
                continue
            pages_read += 1
            if len(text) > char_budget:
                chunks.append(text[:char_budget])
                char_budget = 0
            else:
                chunks.append(text)
                char_budget -= len(text)
        total_pages = doc.page_count
    finally:
        doc.close()
    full = sanitize_text("\n\n".join(chunks))
    return full, {
        "engine": "pymupdf",
        "total_pages": total_pages,
        "pages_extracted": pages_read,
        "chars_extracted": len(full),
        "truncated": char_budget <= 0 and len(full) >= max_chars,
    }


def _extract_with_pypdf(data: bytes, *, max_chars: int) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return "", {"skipped": "no-pdf-engine"}
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        return "", {"engine": "pypdf", "skipped": "pdf-parse-failed", "error": str(exc)[:200]}
    chunks: list[str] = []
    char_budget = max_chars
    pages_read = 0
    for page in reader.pages:
        if char_budget <= 0:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text:
            continue
        pages_read += 1
        if len(text) > char_budget:
            chunks.append(text[:char_budget])
            char_budget = 0
        else:
            chunks.append(text)
            char_budget -= len(text)
    full = sanitize_text("\n\n".join(chunks))
    return full, {
        "engine": "pypdf",
        "total_pages": len(reader.pages),
        "pages_extracted": pages_read,
        "chars_extracted": len(full),
        "truncated": char_budget <= 0 and len(full) >= max_chars,
    }
