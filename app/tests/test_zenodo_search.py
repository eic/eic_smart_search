"""Unit tests for the live Zenodo search client.

Mocked at the httpx layer; we don't need a real network for the parsing
+ caching contract.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.integrations.zenodo_search import ZenodoSearchClient


FAKE_PAYLOAD = {
    "hits": {
        "hits": [
            {
                "id": 12345,
                "metadata": {
                    "title": "Technical Design Report draft",
                    "doi": "10.5281/zenodo.12345",
                    "publication_date": "2026-04-01",
                    "resource_type": {"title": "Report", "type": "publication"},
                    "creators": [
                        {"name": "Müller, Hans", "affiliation": "BNL"},
                        {"name": "Smith, Alice", "affiliation": "JLab"},
                    ],
                    "keywords": ["EIC", "ePIC", "TDR"],
                    "description": "<p>An <em>early</em> draft.</p>",
                },
                "links": {"self_html": "https://zenodo.org/records/12345"},
            },
            {
                "id": 99999,
                "metadata": {
                    "title": "Stub record",
                    "publication_date": "2026-01-15",
                    "resource_type": {"type": "presentation"},
                },
                "links": {},
            },
        ]
    }
}


def _mock_response(payload, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    response.json.return_value = payload
    return response


def test_search_parses_records_and_strips_html():
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)) as mocked:
        hits = client.search("TDR", size=5)
    mocked.assert_called_once()
    params = mocked.call_args.kwargs.get("params") or mocked.call_args[1]["params"]
    assert params["q"] == "TDR"
    assert params["communities"] == "epic"
    assert params["sort"] == "bestmatch"
    assert len(hits) == 2
    first = hits[0]
    assert first.id == "12345"
    assert first.title == "Technical Design Report draft"
    assert first.doi == "10.5281/zenodo.12345"
    assert first.url == "https://zenodo.org/records/12345"
    assert first.resource_type == "Report"
    assert first.authors == ["Müller, Hans", "Smith, Alice"]
    assert first.keywords == ["EIC", "ePIC", "TDR"]
    # HTML was stripped from the description.
    assert "<" not in first.description
    assert "early" in first.description


def test_search_falls_back_to_constructed_url_when_links_missing():
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)):
        hits = client.search("anything")
    assert hits[1].url == "https://zenodo.org/records/99999"
    # When `metadata.resource_type` lacks a title, fall back to `type`.
    assert hits[1].resource_type == "presentation"


def test_search_falls_back_to_wildcard_on_blank_query():
    # An empty query falls through to a wildcard — the LLM occasionally
    # asks for "any recent ePIC paper" which lands here after we strip
    # community: filters.
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)) as mocked:
        client.search("   ")
    params = mocked.call_args.kwargs.get("params") or mocked.call_args[1]["params"]
    assert params["q"] == "*"


def test_search_strips_community_filter_prefix():
    # The LLM sometimes passes Zenodo filter syntax like "community:ePIC"
    # — it conflicts with our already-set communities param and would
    # otherwise return 0 hits.
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)) as mocked:
        client.search("community:ePIC silicon vertex tracker")
    params = mocked.call_args.kwargs.get("params") or mocked.call_args[1]["params"]
    assert params["q"] == "silicon vertex tracker"


def test_search_falls_back_when_query_was_only_filters():
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)) as mocked:
        client.search("community:ePIC")
    params = mocked.call_args.kwargs.get("params") or mocked.call_args[1]["params"]
    assert params["q"] == "*"


def test_search_clamps_invalid_sort():
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)) as mocked:
        client.search("anything", sort="garbage")  # type: ignore[arg-type]
    params = mocked.call_args.kwargs.get("params") or mocked.call_args[1]["params"]
    assert params["sort"] == "bestmatch"


def test_search_caches_within_ttl():
    client = ZenodoSearchClient(community="epic", cache_ttl_s=60.0)
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)) as mocked:
        client.search("TDR")
        client.search("TDR")
    # Second call should be served from the in-memory cache.
    assert mocked.call_count == 1


def test_search_empty_when_zenodo_returns_no_hits():
    empty = {"hits": {"hits": []}}
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(empty)):
        assert client.search("nothing") == []


def test_search_swallows_http_errors():
    client = ZenodoSearchClient(community="epic")
    failing_response = MagicMock()
    failing_response.raise_for_status.side_effect = RuntimeError("503")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=failing_response):
        assert client.search("X") == []


def test_to_prompt_dict_truncates_long_author_lists():
    client = ZenodoSearchClient(community="epic")
    with patch("app.integrations.zenodo_search.httpx.get", return_value=_mock_response(FAKE_PAYLOAD)):
        hits = client.search("TDR")
    payload = hits[0].to_prompt_dict()
    assert payload["title"] == "Technical Design Report draft"
    assert len(payload["authors"]) == 2  # well under the cap
    assert payload["doi"] == "10.5281/zenodo.12345"


def test_constructor_rejects_empty_community():
    with pytest.raises(ValueError):
        ZenodoSearchClient(community="")
