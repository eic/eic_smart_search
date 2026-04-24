from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.parent_expand import ParentExpansionConfig, expand_parents
from app.retrieval.types import RetrievedChunk


@dataclass
class _StubChunkRow:
    document_id: str
    chunk_index: int
    content: str
    heading_path: list[str]
    token_count: int = 100


class _StubScalars:
    def __init__(self, rows: list[_StubChunkRow]) -> None:
        self._rows = rows

    def all(self) -> list[_StubChunkRow]:
        return self._rows


class _StubDb:
    """Minimal Session-like stub that returns preloaded chunk rows keyed by document_id."""

    def __init__(self, rows_by_doc: dict[str, list[_StubChunkRow]]) -> None:
        self._rows_by_doc = rows_by_doc
        self.queries: list[str] = []

    def scalars(self, stmt):  # noqa: ANN001 — accepts SQLAlchemy Select
        doc_ids = self._extract_doc_ids(stmt)
        rows: list[_StubChunkRow] = []
        for doc_id in doc_ids:
            rows.extend(self._rows_by_doc.get(doc_id, []))
            self.queries.append(doc_id)
        return _StubScalars(rows)

    @staticmethod
    def _extract_doc_ids(stmt) -> list[str]:  # noqa: ANN001
        # The select has a WHERE Chunk.document_id == <value>; read it off the compile.
        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
        text = str(compiled)
        # naive parse: "WHERE chunks.document_id = 'doc-a'"
        import re

        match = re.search(r"document_id\s*=\s*'([^']+)'", text)
        return [match.group(1)] if match else []


def _hit(
    chunk_id: str,
    document_id: str,
    score: float,
    heading_path: list[str],
    filetype: str | None = "md",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_name="repo",
        source_type="github_repo",
        title="doc",
        url="https://example.com/doc.md",
        content="original short content",
        score=score,
        heading_path=heading_path,
        filetype=filetype,
    )


def test_disabled_returns_chunks_unchanged() -> None:
    config = ParentExpansionConfig(enabled=False, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=1200)
    hits = [_hit("c1", "doc-a", 0.9, ["H1", "H2"])]

    out, debug = expand_parents(_StubDb({}), hits, config)

    assert out == hits
    assert debug["enabled"] is False
    assert debug["expanded"] == 0


def test_non_markdown_passes_through() -> None:
    config = ParentExpansionConfig(enabled=True, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=1200)
    hits = [_hit("c1", "doc-a", 0.9, ["H1"], filetype="py")]

    out, debug = expand_parents(_StubDb({}), hits, config)

    assert out[0].content == "original short content"
    assert debug["expanded"] == 0


def test_stitches_siblings_sharing_heading_prefix() -> None:
    config = ParentExpansionConfig(enabled=True, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=1200)
    rows = [
        _StubChunkRow("doc-a", 0, "Intro paragraph.", ["Getting Started", "Install", "Docker"], 10),
        _StubChunkRow("doc-a", 1, "Run docker compose up.", ["Getting Started", "Install", "Docker"], 10),
        _StubChunkRow("doc-a", 2, "Verify the app is up.", ["Getting Started", "Install", "Verify"], 10),
        _StubChunkRow("doc-a", 3, "A totally different section.", ["Reference", "API"], 10),
    ]
    db = _StubDb({"doc-a": rows})
    hits = [_hit("c0", "doc-a", 0.9, ["Getting Started", "Install", "Docker"])]

    out, debug = expand_parents(db, hits, config)

    # depth=2 → parent key is ("Getting Started", "Install"); rows 0,1,2 match; row 3 does not.
    assert "Intro paragraph." in out[0].content
    assert "Run docker compose up." in out[0].content
    assert "Verify the app is up." in out[0].content
    assert "totally different" not in out[0].content
    assert out[0].metadata["parent_expansion"]["merged_siblings"] == 3
    assert out[0].metadata["parent_expansion"]["parent_key"] == ["Getting Started", "Install"]
    assert debug["expanded"] == 1


def test_deduplicates_hits_sharing_same_parent() -> None:
    config = ParentExpansionConfig(enabled=True, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=1200)
    rows = [
        _StubChunkRow("doc-a", 0, "Section intro.", ["H1", "H2"], 10),
        _StubChunkRow("doc-a", 1, "More detail.", ["H1", "H2"], 10),
    ]
    db = _StubDb({"doc-a": rows})
    # Two hits from the same document sharing H1+H2 — the lower-scoring one should be dropped.
    hits = [
        _hit("c0", "doc-a", 0.9, ["H1", "H2"]),
        _hit("c1", "doc-a", 0.4, ["H1", "H2"]),
    ]

    out, debug = expand_parents(db, hits, config)

    assert len(out) == 1
    assert out[0].chunk_id == "c0"
    assert debug["expanded"] == 1
    assert debug["merged_duplicates"] == 1


def test_respects_max_tokens_budget() -> None:
    config = ParentExpansionConfig(enabled=True, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=25)
    rows = [
        _StubChunkRow("doc-a", 0, "first", ["H1", "H2"], 10),
        _StubChunkRow("doc-a", 1, "second", ["H1", "H2"], 10),
        _StubChunkRow("doc-a", 2, "third", ["H1", "H2"], 10),
    ]
    db = _StubDb({"doc-a": rows})
    hits = [_hit("c0", "doc-a", 0.9, ["H1", "H2"])]

    out, _ = expand_parents(db, hits, config)

    # 10 + 10 = 20 fits; 20 + 10 > 25 stops before including the third chunk.
    assert "first" in out[0].content
    assert "second" in out[0].content
    assert "third" not in out[0].content


def test_empty_heading_path_expands_whole_document_capped() -> None:
    config = ParentExpansionConfig(enabled=True, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=1200)
    rows = [
        _StubChunkRow("doc-a", 0, "alpha", [], 10),
        _StubChunkRow("doc-a", 1, "bravo", [], 10),
    ]
    db = _StubDb({"doc-a": rows})
    hits = [_hit("c0", "doc-a", 0.9, [])]

    out, debug = expand_parents(db, hits, config)

    assert "alpha" in out[0].content
    assert "bravo" in out[0].content
    assert debug["expanded"] == 1


def test_filetype_normalized_dot_and_case() -> None:
    config = ParentExpansionConfig(enabled=True, filetypes=frozenset({"md"}), heading_depth=2, max_tokens=1200)
    rows = [_StubChunkRow("doc-a", 0, "body", ["H1", "H2"], 10)]
    db = _StubDb({"doc-a": rows})
    hits = [_hit("c0", "doc-a", 0.9, ["H1", "H2"], filetype=".MD")]

    out, debug = expand_parents(db, hits, config)

    assert debug["expanded"] == 1
    assert "body" in out[0].content
