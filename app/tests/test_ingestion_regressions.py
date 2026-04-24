"""Regression tests for bugs hit during multi-source rollout.

These test the raw primitives that caused production-like failures:
  * NUL / control bytes in PDF-extracted text crashing Postgres
  * YAML front-matter dates breaking JSONB serialization
  * MediaWiki query-form namespace URLs slipping past the path-prefix filter
  * WebsiteCrawler refusing to follow links when content is below the
    25-word threshold (caused the whole tutorial subtree to be missed)
"""
from __future__ import annotations

import datetime as dt

import pytest


# ---------------------------------------------------------------------------
# NUL-byte sanitizer (Zenodo)
# ---------------------------------------------------------------------------

def test_zenodo_sanitize_strips_nul_and_control_chars() -> None:
    from app.ingestion.zenodo import _sanitize_text

    raw = "clean text\x00with\x01control\x1fbytes\tand\na tab plus newline"
    out = _sanitize_text(raw)
    assert "\x00" not in out
    assert "\x01" not in out
    assert "\x1f" not in out
    # Tab and newline should survive — they're not in the strip range.
    assert "\t" in out
    assert "\n" in out


def test_zenodo_sanitize_noop_on_plain_text() -> None:
    from app.ingestion.zenodo import _sanitize_text

    assert _sanitize_text("hello world") == "hello world"
    assert _sanitize_text("") == ""


# ---------------------------------------------------------------------------
# GitHub-org YAML front-matter date -> JSON-safe
# ---------------------------------------------------------------------------

def test_github_org_json_safe_serializes_date_and_datetime() -> None:
    from app.ingestion.github_org import _json_safe

    value = {
        "date": dt.date(2026, 4, 1),
        "ts": dt.datetime(2026, 4, 1, 12, 30, 45),
        "nested": {"when": dt.date(2025, 1, 1)},
        "list_of_dates": [dt.date(2024, 1, 1), dt.date(2024, 2, 1)],
        "plain": "hello",
        "count": 42,
    }
    out = _json_safe(value)
    import json
    # Must round-trip through json.dumps without raising.
    json.dumps(out)
    assert out["date"] == "2026-04-01"
    assert out["ts"].startswith("2026-04-01T12:30:45")
    assert out["nested"]["when"] == "2025-01-01"
    assert out["list_of_dates"] == ["2024-01-01", "2024-02-01"]
    assert out["plain"] == "hello"
    assert out["count"] == 42


# ---------------------------------------------------------------------------
# WebsiteCrawler URL canonicalization + MediaWiki filter
# ---------------------------------------------------------------------------

def _make_crawler(**kwargs):
    """Build a WebsiteCrawler without real settings (we only need _canonicalize)."""
    from app.ingestion.web import WebsiteCrawler

    crawler = WebsiteCrawler.__new__(WebsiteCrawler)
    crawler.allowed_netloc = kwargs.get("allowed_netloc", "example.com")
    crawler.allowed_path_prefix = kwargs.get("allowed_path_prefix")
    if crawler.allowed_path_prefix and not crawler.allowed_path_prefix.endswith("/"):
        crawler.allowed_path_prefix = crawler.allowed_path_prefix.rstrip("/") + "/"
    from app.ingestion.web import _parse_prefixes

    crawler.exclude_prefixes = _parse_prefixes(kwargs.get("exclude_prefixes"))
    return crawler


def test_canonicalize_rejects_cross_origin() -> None:
    crawler = _make_crawler(allowed_netloc="example.com")
    assert crawler._canonicalize("https://evil.com/page") is None


def test_canonicalize_rejects_binary_extensions() -> None:
    crawler = _make_crawler(allowed_netloc="example.com")
    for url in [
        "https://example.com/file.pdf",
        "https://example.com/a/photo.JPG",
        "https://example.com/geometry.xml",
        "https://example.com/mesh.obj",
        "https://example.com/bundle.js",
    ]:
        assert crawler._canonicalize(url) is None, url


def test_canonicalize_exclude_prefix_hit() -> None:
    crawler = _make_crawler(
        allowed_netloc="eic.github.io",
        exclude_prefixes="/epic/artifacts,/EDM4eic/",
    )
    assert crawler._canonicalize("https://eic.github.io/epic/artifacts/foo") is None
    assert crawler._canonicalize("https://eic.github.io/EDM4eic/cls.html") is None
    assert crawler._canonicalize("https://eic.github.io/docs/intro.html") == "https://eic.github.io/docs/intro.html"


def test_canonicalize_normalizes_double_slashes() -> None:
    crawler = _make_crawler(allowed_netloc="eic.github.io", exclude_prefixes="/epic/artifacts")
    # Double-slash path should normalize *and* be filtered by exclusion.
    assert crawler._canonicalize("https://eic.github.io/epic//artifacts/blob") is None


def test_canonicalize_allowed_path_prefix_keeps_bnl_in_scope() -> None:
    crawler = _make_crawler(
        allowed_netloc="wiki.bnl.gov",
        allowed_path_prefix="/EPIC/",
    )
    assert crawler._canonicalize("https://wiki.bnl.gov/EPIC/index.php?title=Main_Page") is not None
    # /conferences/ is on the same host but outside the EPIC wiki tree.
    assert crawler._canonicalize("https://wiki.bnl.gov/conferences/index.php/SomePage") is None


def test_canonicalize_mediawiki_query_form_namespace_rejected() -> None:
    crawler = _make_crawler(
        allowed_netloc="wiki.bnl.gov",
        allowed_path_prefix="/EPIC/",
    )
    # Pretty path form would be caught by exclude_prefixes; query form needs
    # title= parsing.
    for bad in [
        "https://wiki.bnl.gov/EPIC/index.php?title=Special:CiteThisPage",
        "https://wiki.bnl.gov/EPIC/index.php?title=User:Someone",
        "https://wiki.bnl.gov/EPIC/index.php?title=Talk:Main_Page",
        "https://wiki.bnl.gov/EPIC/index.php?title=File:Diagram.png",
        # URL-encoded colon should also be caught.
        "https://wiki.bnl.gov/EPIC/index.php?title=Special%3ARecentChanges",
    ]:
        assert crawler._canonicalize(bad) is None, bad


def test_canonicalize_mediawiki_action_params_rejected() -> None:
    crawler = _make_crawler(allowed_netloc="wiki.bnl.gov", allowed_path_prefix="/EPIC/")
    for bad in [
        "https://wiki.bnl.gov/EPIC/index.php?title=Main_Page&action=edit",
        "https://wiki.bnl.gov/EPIC/index.php?title=Main_Page&action=history",
        "https://wiki.bnl.gov/EPIC/index.php?title=Main_Page&action=info",
        "https://wiki.bnl.gov/EPIC/index.php?title=Main_Page&oldid=12345",
        "https://wiki.bnl.gov/EPIC/index.php?title=Main_Page&mobileaction=toggle_view_mobile",
    ]:
        assert crawler._canonicalize(bad) is None, bad
    # A plain content URL on the same page should pass.
    assert crawler._canonicalize(
        "https://wiki.bnl.gov/EPIC/index.php?title=Tracking"
    ) is not None


# ---------------------------------------------------------------------------
# Zenodo authors/keywords helpers
# ---------------------------------------------------------------------------

def test_zenodo_authors_formatter() -> None:
    from app.ingestion.zenodo import _authors

    creators = [
        {"name": "Alice", "affiliation": "BNL"},
        {"name": "Bob"},
        {"name": "  ", "affiliation": "skipped"},
        {},
    ]
    out = _authors(creators)
    assert out == ["Alice (BNL)", "Bob"]


def test_zenodo_keywords_formatter_filters_nonstring_and_blanks() -> None:
    from app.ingestion.zenodo import _keywords

    out = _keywords(["rucio", "", "  ", None, 42, "detector"])
    assert out == ["rucio", "detector"]


def test_zenodo_pdf_detection_by_mimetype_and_key() -> None:
    from app.ingestion.zenodo import ZenodoConnector

    # Build without calling __init__ so we don't need real settings.
    conn = ZenodoConnector.__new__(ZenodoConnector)
    assert conn._is_pdf({"key": "report.PDF"}) is True
    assert conn._is_pdf({"key": "deck.pptx", "mimetype": "application/pdf"}) is True
    assert conn._is_pdf({"key": "slides.pptx"}) is False
    assert conn._is_pdf({}) is False


# ---------------------------------------------------------------------------
# EpicInternalCrawler: source_name must survive through the parent __init__
# ---------------------------------------------------------------------------

def test_epic_internal_keeps_its_source_name(monkeypatch) -> None:
    """Regression: a previous refactor of WebsiteCrawler.__init__ silently
    overwrote subclass source_name with the parent default 'eic_website'.
    """
    from app.core.config import Settings
    from app.ingestion.internal import EpicInternalCrawler

    # Build a Settings instance with only the fields the crawler touches.
    settings = Settings(
        _env_file=None,
        EPIC_INTERNAL_START_URL="https://www.epic-eic.org/index-internal.html",
        EPIC_INTERNAL_MAX_PAGES=100,
    )
    crawler = EpicInternalCrawler(settings)
    assert crawler.source_name == "epic_internal"
    assert crawler.source_type == "internal_website"
    assert crawler.visibility == "internal"


# ---------------------------------------------------------------------------
# Website crawler: short-content pages must still extract outgoing links
# ---------------------------------------------------------------------------

def test_short_content_page_still_queues_links(monkeypatch) -> None:
    """Regression: when an HTML page extracted <25 words of text, the crawler
    used to `continue` past link extraction, stranding its children in the
    queue. This broke BFS through thin index pages like
    documentation/tutorials.html which had few words but many outbound links.
    """
    from unittest.mock import MagicMock

    from app.core.config import Settings
    from app.ingestion.html import ExtractedHtml
    from app.ingestion.web import WebsiteCrawler

    settings = Settings(
        _env_file=None,
        EIC_SITE_URL="https://example.com/",
        EIC_SITE_MAX_PAGES=10,
        EIC_SITE_EXCLUDE_PREFIXES="",
    )

    def fake_extract(body, url, last_modified):
        if url.endswith("/index"):
            # Short content, many outbound links.
            return ExtractedHtml(
                title="Index",
                content="just a few words here only",  # <25 words
                headings=[],
                breadcrumbs=[],
                links=[
                    "https://example.com/one",
                    "https://example.com/two",
                ],
                last_updated=None,
            )
        return ExtractedHtml(
            title=url.rsplit("/", 1)[-1],
            content="x " * 30,  # >25 words
            headings=[],
            breadcrumbs=[],
            links=[],
            last_updated=None,
        )

    # Fake httpx client: every GET returns an HTML body.
    class _FakeResp:
        def __init__(self, url):
            self.status_code = 200
            self.text = ""
            self.headers = {"content-type": "text/html; charset=utf-8"}
            self.url = url

        def raise_for_status(self) -> None:  # noqa: D401
            return None

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url):
            return _FakeResp(url)

    monkeypatch.setattr("app.ingestion.web.extract_html", fake_extract)
    monkeypatch.setattr("app.ingestion.web.httpx.Client", _FakeClient)

    crawler = WebsiteCrawler(
        settings,
        source_name="t",
        start_url="https://example.com/index",
        max_pages=5,
        exclude_prefixes="",
    )
    docs = crawler.iter_documents()
    urls = {doc.url for doc in docs}
    # The short-content root itself should NOT be in documents.
    assert "https://example.com/index" not in urls
    # But its outbound links must have been crawled.
    assert "https://example.com/one" in urls
    assert "https://example.com/two" in urls
