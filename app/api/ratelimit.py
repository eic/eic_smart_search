"""Per-client-IP sliding-window rate limiting for the public query endpoints.

In-process (matches the single-container deployment). Sliding window of
one minute; over-limit requests get 429 with Retry-After. Client identity
prefers the first X-Forwarded-For hop (the stack sits behind a tunnel).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings

_WINDOW_S = 60.0
_lock = threading.Lock()
_hits: dict[str, deque[float]] = {}
_MAX_CLIENTS = 10_000  # memory backstop


def _client_ip(request: Request) -> str:
    # Behind cloudflared, CF-Connecting-IP is the real client and cannot be
    # spoofed by the client (Cloudflare sets it). X-Forwarded-For hop 0 is
    # attacker-controlled — using it lets one caller mint unlimited buckets;
    # take the LAST hop (added by our own proxy) instead.
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request, limit: int) -> None:
    """Raise 429 when `limit` requests/minute is exceeded for this client."""
    if limit <= 0:
        return
    ip = _client_ip(request)
    now = time.monotonic()
    with _lock:
        window = _hits.get(ip)
        if window is None:
            if len(_hits) >= _MAX_CLIENTS:
                _hits.clear()  # crude but bounded; legitimate windows rebuild in seconds
            window = _hits[ip] = deque()
        cutoff = now - _WINDOW_S
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit:
            retry_after = max(1, int(window[0] + _WINDOW_S - now) + 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)


def _settings_dep() -> Settings:
    return get_settings()


def query_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
) -> None:
    check_rate_limit(request, int(getattr(settings, "QUERY_RATE_LIMIT_PER_MINUTE", 0)))


RateLimited = Annotated[None, Depends(query_rate_limit)]
