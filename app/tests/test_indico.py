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
    client = IndicoClient("https://indico.bnl.gov/category/402/", cache_ttl_s=60, warm_on_init=False)
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


def test_split_window_slices_wide_ranges():
    from datetime import date, timedelta

    from app.integrations.indico import IndicoClient

    spans = IndicoClient._split_window("-180d", "60d")
    # 240 days at 60-day slices -> 4 spans
    assert len(spans) == 4
    first_from = date.fromisoformat(spans[0][0])
    last_to = date.fromisoformat(spans[-1][1])
    assert first_from == date.today() + timedelta(days=-180)
    assert last_to == date.today() + timedelta(days=60)
    # Contiguous: each span starts where the previous ended.
    for (_, prev_to), (next_from, _) in zip(spans, spans[1:]):
        assert prev_to == next_from


def test_split_window_passthrough_on_unparseable():
    from app.integrations.indico import IndicoClient

    assert IndicoClient._split_window("yesterday", "30d") == [("yesterday", "30d")]
    # Narrow windows stay single-span.
    assert len(IndicoClient._split_window("-10d", "10d")) == 1


# ---- stale-while-revalidate + warm() -------------------------------------

URL = "https://indico.bnl.gov/category/402/"


def _sync_client(**kwargs) -> tuple[IndicoClient, list]:
    """Client whose background threads run inline; records each spawn."""
    client = IndicoClient(URL, cache_ttl_s=60, warm_on_init=False, **kwargs)
    spawned: list = []

    def _inline(target, *args):
        spawned.append((target, args))
        target(*args)

    client._spawn = _inline  # type: ignore[method-assign]
    return client, spawned


def _expire_all(client: IndicoClient) -> None:
    for entry in client._cache.values():
        entry.expires_at = 0.0


def test_expired_entry_served_stale_and_refreshed_in_background():
    client = IndicoClient(URL, cache_ttl_s=60, warm_on_init=False)
    spawned: list = []
    client._spawn = lambda target, *args: spawned.append((target, args))  # type: ignore[method-assign]
    with patch("app.integrations.indico.httpx.get", _mock_httpx_get()):
        assert len(client.search()) == 2
    _expire_all(client)
    newer = {"results": [dict(FAKE_PAYLOAD["results"][0], id=1, title="New DIRC meeting")]}
    mock = _mock_httpx_get(newer)
    with patch("app.integrations.indico.httpx.get", mock):
        events = client.search()
        # Stale data returned with no network call on the caller's path.
        assert [e.id for e in events] == ["31239", "32601"]
        assert mock.call_count == 0
        assert len(spawned) == 1
        # A second call while the refresh is in flight doesn't spawn another.
        client.search()
        assert len(spawned) == 1
        target, args = spawned[0]
        target(*args)  # run the background refresh now
    assert mock.call_count == 1
    with patch("app.integrations.indico.httpx.get", side_effect=AssertionError("no fetch")):
        assert [e.title for e in client.search()] == ["New DIRC meeting"]


def test_failed_refresh_keeps_stale_data_and_warns_once(caplog):
    client, _ = _sync_client()
    with patch("app.integrations.indico.httpx.get", _mock_httpx_get()):
        client.search()
    with patch("app.integrations.indico.httpx.get", side_effect=TimeoutError("The read operation timed out")):
        with caplog.at_level("WARNING", logger="app.integrations.indico"):
            for _ in range(3):
                _expire_all(client)
                assert [e.id for e in client.search()] == ["31239", "32601"]
    stale_warnings = [r for r in caplog.records if r.getMessage() == "indico_refresh_failed_serving_stale"]
    assert len(stale_warnings) == 1
    # Retry is paced, not immediate.
    assert all(e.expires_at > 0 for e in client._cache.values())


def test_search_blocks_on_at_most_one_fetch_round():
    client = IndicoClient(URL, cache_ttl_s=60, warm_on_init=False)
    spawned: list = []
    client._spawn = lambda target, *args: spawned.append((target, args))  # type: ignore[method-assign]
    mock = _mock_httpx_get({"results": []})
    with patch("app.integrations.indico.httpx.get", mock):
        assert client.search(query="DIRC") == []
    # Primary window fetched synchronously; the 365d fallback deferred.
    assert mock.call_count == 1
    assert len(spawned) == 1


def test_warm_fetches_default_windows():
    client = IndicoClient(URL, cache_ttl_s=60, warm_on_init=False)
    mock = _mock_httpx_get()
    with patch("app.integrations.indico.httpx.get", mock):
        assert client.warm() == len(IndicoClient.DEFAULT_WARM_WINDOWS)
        # Everything fresh now: a second warm() is a no-op.
        assert client.warm() == 0
        client.search(from_date="today", to_date="30d")
        client.search()
    requested = {(c.kwargs["params"]["from"], c.kwargs["params"]["to"]) for c in mock.call_args_list}
    assert requested == set(IndicoClient.DEFAULT_WARM_WINDOWS)
    assert ("today", "365d") in requested and ("-7d", "30d") in requested
    assert mock.call_count == len(IndicoClient.DEFAULT_WARM_WINDOWS)


def test_constructor_warms_in_background_thread():
    spawned: list = []
    with patch.object(IndicoClient, "_spawn", lambda self, target, *args: spawned.append(target)):
        client = IndicoClient(URL, cache_ttl_s=60)
        IndicoClient(URL, cache_ttl_s=0)
    assert spawned == [client.warm]
