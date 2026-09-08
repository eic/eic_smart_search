from __future__ import annotations

import httpx
import pytest

from app.integrations.code_search import CodeSearchClient


def _client() -> CodeSearchClient:
    return CodeSearchClient("eic", token=None, cache_ttl_s=0)


def test_search_rejects_empty_query() -> None:
    assert _client().search_code("") == {"error": "empty query"}


def test_search_scopes_to_org(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["q"] = params["q"]
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "repository": {"full_name": "eic/EICrecon"},
                        "path": "src/x.h",
                        "html_url": "https://github.com/eic/EICrecon/blob/main/src/x.h",
                        "text_matches": [{"fragment": "int  PID  = 90;"}],
                    }
                ]
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    result = _client().search_code("PID type 90")
    assert "org:eic" in captured["q"]
    assert result["count"] == 1
    assert result["hits"][0]["repo"] == "eic/EICrecon"
    assert result["hits"][0]["fragments"] == ["int PID = 90;"]


def test_search_foreign_repo_falls_back_to_org(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["q"] = params["q"]
        return httpx.Response(200, json={"items": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    _client().search_code("x", repo="evil/repo")
    assert "org:eic" in captured["q"] and "evil" not in captured["q"]


def test_search_reports_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url, params=None, headers=None, timeout=None):
        return httpx.Response(403, json={}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    result = _client().search_code("anything")
    assert "rate limit" in result["error"]


def test_fetch_rejects_foreign_repo_and_bad_path() -> None:
    client = _client()
    assert "error" in client.fetch_code_file("evil/repo", "x.h")
    assert "error" in client.fetch_code_file("eic/EICrecon", "../etc/passwd")
    assert "error" in client.fetch_code_file("eic/EICrecon/extra", "x.h")


def test_fetch_windows_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    text = "\n".join(f"line{i}" for i in range(1, 1001))

    def fake_safe_get(http_client, url):
        return httpx.Response(200, text=text, request=httpx.Request("GET", url))

    monkeypatch.setattr("app.integrations.code_search.safe_get", fake_safe_get)
    result = client.fetch_code_file("eic/EICrecon", "src/big.cc", start_line=500, end_line=502)
    assert result["start_line"] == 500 and result["end_line"] == 502
    assert result["content"] == "500: line500\n501: line501\n502: line502"
    assert result["total_lines"] == 1000
    assert result["url"].endswith("#L500")
    # Unbounded request clips to the 400-line window.
    result = client.fetch_code_file("eic/EICrecon", "src/big.cc")
    assert result["end_line"] == 400
