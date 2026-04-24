from unittest.mock import MagicMock, patch

import pytest

from app.integrations.indico import IndicoClient, _clean_html, _combine_dt


def test_combine_dt_formats_timezone_tail():
    assert _combine_dt({"date": "2026-05-22", "time": "11:00:00", "tz": "America/New_York"}) == (
        "2026-05-22T11:00:00 America/New_York"
    )


def test_combine_dt_handles_missing_time():
    assert _combine_dt({"date": "2026-05-22"}) == "2026-05-22"


def test_combine_dt_handles_none():
    assert _combine_dt(None) == ""


def test_clean_html_strips_tags_and_collapses_whitespace():
    raw = "<p>Zoom:\n  <a href='x'>join</a></p>\n<br>Room 1-108"
    assert _clean_html(raw) == "Zoom: join Room 1-108"


def test_to_export_url_rewrites_category_path():
    assert IndicoClient._to_export_url("https://indico.bnl.gov/category/402/") == (
        "https://indico.bnl.gov/export/categ/402.json"
    )
    with pytest.raises(ValueError):
        IndicoClient._to_export_url("https://indico.bnl.gov/event/32601")


FAKE_PAYLOAD = {
    "count": 2,
    "results": [
        {
            "id": 32601,
            "title": "DIRC Working Group meeting",
            "url": "https://indico.bnl.gov/event/32601/",
            "category": "hpDIRC",
            "room": "Bldg 510",
            "location": "",
            "description": "<p>Zoom: https://x</p>",
            "startDate": {"date": "2026-05-22", "time": "11:00:00", "tz": "America/New_York"},
            "endDate": {"date": "2026-05-22", "time": "13:00:00", "tz": "America/New_York"},
        },
        {
            "id": 31239,
            "title": "SVT Mechanics",
            "url": "https://indico.bnl.gov/event/31239/",
            "category": "SVT",
            "room": "",
            "location": "",
            "description": "",
            "startDate": {"date": "2026-05-20", "time": "11:30:00", "tz": "America/New_York"},
            "endDate": {"date": "2026-05-20", "time": "12:25:00", "tz": "America/New_York"},
        },
    ],
}


def _mock_httpx_get(payload=FAKE_PAYLOAD):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return MagicMock(return_value=response)


def test_search_returns_events_sorted_by_start():
    client = IndicoClient("https://indico.bnl.gov/category/402/", cache_ttl_s=0)
    with patch("app.integrations.indico.httpx.get", _mock_httpx_get()):
        events = client.search()
    assert [e.id for e in events] == ["31239", "32601"]  # earlier start first
    assert events[0].title == "SVT Mechanics"
    assert events[1].description == "Zoom: https://x"


def test_search_query_filters_by_keyword():
    client = IndicoClient("https://indico.bnl.gov/category/402/", cache_ttl_s=0)
    with patch("app.integrations.indico.httpx.get", _mock_httpx_get()):
        events = client.search(query="DIRC")
    assert len(events) == 1
    assert events[0].id == "32601"


def test_search_ignores_api_error_message_payload():
    """Indico returns {'message': '...'} when the date range is malformed;
    we shouldn't pretend we have events when the response is an error hull."""
    client = IndicoClient("https://indico.bnl.gov/category/402/", cache_ttl_s=0)
    with patch("app.integrations.indico.httpx.get", _mock_httpx_get(payload={"message": "bad"})):
        assert client.search() == []


def test_search_caches_window():
    client = IndicoClient("https://indico.bnl.gov/category/402/", cache_ttl_s=60)
    mock = _mock_httpx_get()
    with patch("app.integrations.indico.httpx.get", mock):
        client.search()
        client.search()
        client.search()
    # One network call — subsequent search() calls hit the in-memory cache.
    assert mock.call_count == 1


def test_search_network_error_returns_empty():
    client = IndicoClient("https://indico.bnl.gov/category/402/", cache_ttl_s=0)
    with patch("app.integrations.indico.httpx.get", side_effect=ConnectionError("boom")):
        assert client.search() == []
