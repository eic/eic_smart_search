"""Shared GitHub REST helpers for the issue/release connectors."""
from __future__ import annotations

import logging

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"


def github_headers(settings: Settings) -> dict[str, str]:
    """REST headers, dropping a broken GITHUB_TOKEN instead of failing.

    An expired/revoked token turns every request into a 401 — worse than
    anonymous access (60 req/h but working). Validate once against
    /rate_limit and fall back to anonymous with a loud warning.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": settings.INGEST_USER_AGENT,
    }
    token = settings.GITHUB_TOKEN
    if not token:
        return headers
    authed = {**headers, "Authorization": f"Bearer {token}"}
    try:
        response = httpx.get(f"{API_BASE}/rate_limit", headers=authed, timeout=10)
    except httpx.HTTPError as exc:
        logger.warning("github_token_check_failed", extra={"error": str(exc)[:200]})
        return authed  # network hiccup — keep the token, let the caller retry
    if response.status_code == 401:
        logger.warning(
            "github_token_invalid — falling back to anonymous API access "
            "(60 req/h). Fix GITHUB_TOKEN in .env."
        )
        return headers
    return authed
