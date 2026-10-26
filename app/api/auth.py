"""Admin authentication for mutating endpoints (ingest, reindex, eval, …).

Four modes, selected by ``ADMIN_AUTH_MODE``:

  * ``none``       — open (local development default).
  * ``token``      — static shared secret in ``ADMIN_TOKEN``; callers send
                     ``Authorization: Bearer <token>``.
  * ``github_org`` — callers send their own GitHub personal access token
                     (classic PAT with ``read:org``); the API verifies the
                     user is an *active* member of ``ADMIN_GITHUB_ORG``.
  * ``github_sso`` — browser "Sign in with GitHub" OAuth flow. ``/auth/login``
                     redirects to GitHub; the callback checks org membership
                     once and issues an HMAC-signed session cookie. Bearer
                     PATs are still accepted (github_org rules) so curl /
                     Makefile targets keep working without a browser.

Verification results are cached (hash of token → login) for
``ADMIN_AUTH_CACHE_TTL_S`` so repeated calls don't burn GitHub rate limit.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from typing import Annotated, Any

import httpx
from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
SESSION_COOKIE = "eic_admin_session"

_cache_lock = threading.Lock()
# token-sha256 -> (github login, expires_at monotonic)
_membership_cache: dict[str, tuple[str, float]] = {}


def _settings_dep() -> Settings:
    return get_settings()


def _consteq(a: str, b: str) -> bool:
    """Constant-time compare that never raises.

    ``hmac.compare_digest`` on ``str`` raises TypeError for non-ASCII input
    (a 500 for an attacker-supplied cookie/header), so compare UTF-8 bytes.
    """
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except (AttributeError, UnicodeError, TypeError):
        return False


def _well_formed_token(token: str) -> bool:
    """Bearer tokens/PATs are printable ASCII with no whitespace."""
    return bool(token) and all(0x21 <= ord(c) <= 0x7E for c in token)


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _github_login(token: str) -> str | None:
    """Just the authenticated user's login — /user needs no org approval."""
    try:
        resp = httpx.get(
            f"{GITHUB_API}/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "eic-smart-search-auth",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        return str(resp.json().get("login") or "") or None
    except httpx.HTTPError as exc:
        logger.warning("github_login_check_failed", extra={"error": str(exc)[:200]})
        return None


def verify_admin_login(token: str, settings: Settings) -> str | None:
    """Resolve a GitHub token to an authorized admin login, or None.

    Prefers a username allowlist (ADMIN_GITHUB_USERS) — this needs only the
    user's identity, so it works even when the org has OAuth-App access
    restrictions that block org-membership reads. Falls back to the
    active-org-member check when no allowlist is configured.
    """
    allow = getattr(settings, "ADMIN_GITHUB_USERS", "") or ""
    allowed = {u.strip().lower() for u in allow.split(",") if u.strip()}
    if allowed:
        login = _github_login(token)
        if login and login.lower() in allowed:
            return login
        return None
    return _verify_github_org_member(
        token,
        getattr(settings, "ADMIN_GITHUB_ORG", "eic"),
        float(getattr(settings, "ADMIN_AUTH_CACHE_TTL_S", 300.0)),
    )


def _verify_github_org_member(token: str, org: str, cache_ttl_s: float) -> str | None:
    """Return the GitHub login if `token` belongs to an active `org` member."""
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.monotonic()
    with _cache_lock:
        hit = _membership_cache.get(key)
        if hit and hit[1] > now:
            return hit[0]
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "eic-smart-search-auth",
    }
    try:
        user_resp = httpx.get(f"{GITHUB_API}/user", headers=headers, timeout=10)
        if user_resp.status_code != 200:
            return None
        login = str(user_resp.json().get("login") or "")
        if not login:
            return None
        membership_resp = httpx.get(
            f"{GITHUB_API}/user/memberships/orgs/{org}", headers=headers, timeout=10
        )
        if membership_resp.status_code != 200:
            return None
        if membership_resp.json().get("state") != "active":
            return None
    except httpx.HTTPError as exc:
        logger.warning("github_org_auth_check_failed", extra={"error": str(exc)[:200]})
        return None
    with _cache_lock:
        _membership_cache[key] = (login, now + max(cache_ttl_s, 0))
        # Bound the cache — admin callers are few; 256 is generous.
        while len(_membership_cache) > 256:
            _membership_cache.pop(next(iter(_membership_cache)))
    return login


# ---- signed session cookies (github_sso) ---------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_payload(token: str, secret: str) -> dict[str, Any] | None:
    """Verify signature + expiry (`exp`, unix seconds). None on any failure."""
    if not isinstance(token, str) or not _well_formed_token(token):
        return None
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    expected = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    if not _consteq(sig, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or exp < time.time():
        return None
    return payload


def session_login(request: Request, settings: Settings) -> str | None:
    """GitHub login from a valid session cookie, else None."""
    secret = getattr(settings, "ADMIN_SESSION_SECRET", None)
    raw = request.cookies.get(SESSION_COOKIE)
    if not secret or not raw:
        return None
    payload = verify_payload(raw, secret)
    if payload is None:
        return None
    login = payload.get("login")
    return str(login) if login else None


def require_admin(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
) -> str:
    """FastAPI dependency guarding mutating endpoints. Returns the principal."""
    mode = getattr(settings, "ADMIN_AUTH_MODE", "none")
    if mode == "none":
        return "anonymous"
    if mode == "github_sso":
        login = session_login(request, settings)
        if login:
            return f"github:{login}"
        # Fall through to Bearer PAT (github_org rules) for CLI/Make callers.
    token = _bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Sign in at /auth/login, or send Authorization: Bearer <token>"
                if mode == "github_sso"
                else "Authorization: Bearer <token> required"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not _well_formed_token(token):
        # Non-ASCII / whitespace / control chars: can't be a valid secret or
        # GitHub PAT (and would blow up header encoding downstream).
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if mode == "token":
        expected = settings.ADMIN_TOKEN or ""
        if expected and _consteq(token, expected):
            return "admin-token"
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid admin token")
    if mode == "github_sso":
        # Humans use the browser SSO cookie (handled above); automation
        # (CI, cron, Make) sends the shared ADMIN_TOKEN as a service token so
        # it doesn't need a per-user GitHub PAT. Both, or an org PAT below.
        expected = settings.ADMIN_TOKEN or ""
        if expected and _consteq(token, expected):
            return "service-token"
    if mode in ("github_org", "github_sso"):
        login = verify_admin_login(token, settings)
        if login:
            return f"github:{login}"
        if getattr(settings, "ADMIN_GITHUB_USERS", ""):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="GitHub account is not on the admin allowlist (ADMIN_GITHUB_USERS)",
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"GitHub token is not an active member of the "
                f"'{getattr(settings, 'ADMIN_GITHUB_ORG', 'eic')}' organization "
                "(the token needs the read:org scope)"
            ),
        )
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin auth misconfigured")


AdminPrincipal = Annotated[str, Depends(require_admin)]


def is_admin_request(request: Request, settings: Settings) -> bool:
    """Non-raising admin check for request-level privilege flags on public
    endpoints (scope=internal, eval_mode, force_rerank)."""
    try:
        require_admin(request, settings)
        return True
    except HTTPException:
        return False


# ---- OAuth web flow -------------------------------------------------------


def public_base_url(request: Request, settings: Settings) -> str:
    configured = getattr(settings, "PUBLIC_BASE_URL", None)
    if configured:
        return str(configured).rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}"


def oauth_exchange_code(code: str, settings: Settings) -> str | None:
    """Exchange an OAuth authorization code for a user access token."""
    try:
        response = httpx.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": settings.GITHUB_OAUTH_CLIENT_ID,
                "client_secret": settings.GITHUB_OAUTH_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        return str(token) if token else None
    except httpx.HTTPError as exc:
        logger.warning("github_oauth_exchange_failed", extra={"error": str(exc)[:200]})
        return None
