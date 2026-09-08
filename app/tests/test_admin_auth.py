"""Unit tests for the admin-auth dependency (mocked GitHub)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import app.api.auth as auth_mod
from app.api.auth import require_admin


class FakeRequest:
    def __init__(self, token: str | None = None) -> None:
        self.headers = {"authorization": f"Bearer {token}"} if token else {}
        self.cookies: dict[str, str] = {}


class FakeSettings:
    ADMIN_AUTH_MODE = "none"
    ADMIN_TOKEN = None
    ADMIN_GITHUB_ORG = "eic"
    ADMIN_GITHUB_USERS = ""
    ADMIN_AUTH_CACHE_TTL_S = 300.0


def test_mode_none_is_open() -> None:
    assert require_admin(FakeRequest(), FakeSettings()) == "anonymous"


def test_mode_token_accepts_and_rejects() -> None:
    s = FakeSettings()
    s.ADMIN_AUTH_MODE = "token"
    s.ADMIN_TOKEN = "sekret"
    assert require_admin(FakeRequest("sekret"), s) == "admin-token"
    with pytest.raises(HTTPException) as exc:
        require_admin(FakeRequest("wrong"), s)
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        require_admin(FakeRequest(None), s)
    assert exc.value.status_code == 401


def test_mode_github_org_member(monkeypatch) -> None:
    s = FakeSettings()
    s.ADMIN_AUTH_MODE = "github_org"

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if url.endswith("/user"):
            return FakeResp(200, {"login": "aprozo"})
        if "/memberships/orgs/eic" in url:
            return FakeResp(200, {"state": "active"})
        raise AssertionError(url)

    monkeypatch.setattr(auth_mod.httpx, "get", fake_get)
    auth_mod._membership_cache.clear()
    assert require_admin(FakeRequest("ghp_x"), s) == "github:aprozo"
    # Second call hits the cache — no extra HTTP round-trips.
    n = len(calls)
    assert require_admin(FakeRequest("ghp_x"), s) == "github:aprozo"
    assert len(calls) == n


def test_mode_github_org_nonmember_rejected(monkeypatch) -> None:
    s = FakeSettings()
    s.ADMIN_AUTH_MODE = "github_org"

    class FakeResp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/user"):
            return FakeResp(200, {"login": "outsider"})
        return FakeResp(404, {})

    monkeypatch.setattr(auth_mod.httpx, "get", fake_get)
    auth_mod._membership_cache.clear()
    with pytest.raises(HTTPException) as exc:
        require_admin(FakeRequest("ghp_y"), s)
    assert exc.value.status_code == 403


def test_session_cookie_sign_verify_roundtrip() -> None:
    import time

    from app.api.auth import sign_payload, verify_payload

    token = sign_payload({"login": "aprozo", "exp": time.time() + 60}, "secret")
    payload = verify_payload(token, "secret")
    assert payload is not None and payload["login"] == "aprozo"
    # Wrong secret, tampered body, and expired tokens all fail.
    assert verify_payload(token, "other-secret") is None
    body, sig = token.split(".", 1)
    assert verify_payload(body + "x." + sig, "secret") is None
    expired = sign_payload({"login": "aprozo", "exp": time.time() - 1}, "secret")
    assert verify_payload(expired, "secret") is None


def test_mode_github_sso_accepts_session_cookie() -> None:
    import time

    from app.api.auth import SESSION_COOKIE, sign_payload

    s = FakeSettings()
    s.ADMIN_AUTH_MODE = "github_sso"
    s.ADMIN_SESSION_SECRET = "topsecret"

    class CookieRequest(FakeRequest):
        def __init__(self, cookie=None):
            super().__init__(None)
            self.cookies = {SESSION_COOKIE: cookie} if cookie else {}

    good = sign_payload({"login": "aprozo", "exp": time.time() + 60}, "topsecret")
    assert require_admin(CookieRequest(good), s) == "github:aprozo"
    with pytest.raises(HTTPException) as exc:
        require_admin(CookieRequest("garbage.token"), s)
    assert exc.value.status_code == 401


def test_eval_failure_reasons_and_collection() -> None:
    from datetime import datetime

    from app.eval.runner import case_failure_reasons, collect_failures

    passing = {"id": "ok", "retrieval": {"hit": True}, "answer": {"expected_ok": True, "citation_markers_valid": True}}
    failing = {
        "id": "bad", "query": "q", "tags": ["x"],
        "retrieval": {"hit": False, "top_urls": ["u"]},
        "answer": {"expected_ok": False, "missing_terms": ["60"], "citation_markers_valid": True, "text": "wrong"},
        "judge": {"relevance": 4, "faithfulness": 2, "completeness": 3, "comment": "unsupported"},
    }
    assert case_failure_reasons(passing) == []
    assert set(case_failure_reasons(failing)) == {"retrieval-miss", "wrong-answer", "low-faithfulness"}

    class FakeRun:
        def __init__(self, results, created):
            self.id = "r" + str(created)
            self.label = "L"
            self.status = "completed"
            self.results = results
            self.created_at = datetime(2026, 8, created)

    newer = FakeRun([passing | {"id": "bad"}], 9)  # 'bad' now passes
    older = FakeRun([failing], 8)
    failures = collect_failures([newer, older])
    # Newest run wins: 'bad' passed there, so it is off the worklist.
    assert failures == []
    failures = collect_failures([older])
    assert len(failures) == 1 and failures[0]["case_id"] == "bad"
    assert failures[0]["reasons"]


def test_github_sso_accepts_service_token() -> None:
    """In github_sso mode, automation (CI/cron/Make) authenticates with the
    shared ADMIN_TOKEN as a service token — no per-user GitHub PAT needed."""
    s = FakeSettings()
    s.ADMIN_AUTH_MODE = "github_sso"
    s.ADMIN_TOKEN = "svc-secret"
    assert require_admin(FakeRequest("svc-secret"), s) == "service-token"


def test_verify_admin_login_allowlist(monkeypatch) -> None:
    import app.api.auth as auth_mod

    s = FakeSettings()
    s.ADMIN_GITHUB_USERS = "aprozo, someone"
    monkeypatch.setattr(auth_mod, "_github_login", lambda tok: "aprozo")
    assert auth_mod.verify_admin_login("tok", s) == "aprozo"
    monkeypatch.setattr(auth_mod, "_github_login", lambda tok: "outsider")
    assert auth_mod.verify_admin_login("tok", s) is None


def test_github_sso_allowlist_end_to_end(monkeypatch) -> None:
    import app.api.auth as auth_mod

    s = FakeSettings()
    s.ADMIN_AUTH_MODE = "github_sso"
    s.ADMIN_GITHUB_USERS = "aprozo"
    s.ADMIN_TOKEN = None
    monkeypatch.setattr(auth_mod, "_github_login", lambda tok: "aprozo" if tok == "good" else "nope")
    assert require_admin(FakeRequest("good"), s) == "github:aprozo"
    with pytest.raises(HTTPException) as exc:
        require_admin(FakeRequest("bad"), s)
    assert exc.value.status_code == 403
