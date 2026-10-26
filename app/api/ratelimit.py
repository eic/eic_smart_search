"""Per-client-IP sliding-window rate limiting for the public query endpoints.

In-process (matches the single-container deployment). Sliding window of
one minute; over-limit requests get 429 with Retry-After. Client identity
is CF-Connecting-IP, else the last X-Forwarded-For hop (the stack sits
behind a tunnel). IPv6 clients are bucketed per /64 (one subscriber
allocation), so rotating addresses inside a prefix doesn't mint buckets.

Callers of the ``RateLimited`` dependency are also capped at
``MAX_INFLIGHT_PER_CLIENT`` concurrent requests per client key.
"""
from __future__ import annotations

import ipaddress
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings

_WINDOW_S = 60.0
_lock = threading.Lock()
_hits: dict[str, deque[float]] = {}
_MAX_CLIENTS = 10_000  # memory backstop
_IPV6_PREFIX = 64

MAX_INFLIGHT_PER_CLIENT = 3
_inflight: dict[str, int] = {}


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


def _bucket_key(ip: str) -> str:
    """Rate-limit bucket for a client address: IPv4 as-is, IPv6 per /64.

    IPv4-mapped IPv6 (``::ffff:1.2.3.4``) collapses to the IPv4 address.
    Unparseable values (``unknown``, hostnames) are used verbatim.
    """
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]").split("%", 1)[0])
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, _IPV6_PREFIX), strict=False))
    return str(addr)


def _client_key(request: Request) -> str:
    return _bucket_key(_client_ip(request))


def client_key(request: Request) -> str:
    """Public alias: the bucket key used for both rate and in-flight limits."""
    return _client_key(request)


def _evict_locked(now: float) -> None:
    """Make room in ``_hits`` (caller holds ``_lock``).

    First drop clients whose newest hit is outside the window (their state is
    meaningless anyway); if still at the cap, drop the stalest half by most
    recent hit. Active clients keep their windows — unlike a blanket clear,
    an attacker filling the table can't reset everyone else's limit.
    """
    cutoff = now - _WINDOW_S
    for key in [k for k, w in _hits.items() if not w or w[-1] < cutoff]:
        del _hits[key]
    if len(_hits) >= _MAX_CLIENTS:
        by_age = sorted(_hits.items(), key=lambda kv: kv[1][-1])
        for key, _ in by_age[: max(1, len(by_age) // 2)]:
            del _hits[key]


def check_rate_limit(request: Request, limit: int) -> None:
    """Raise 429 when `limit` requests/minute is exceeded for this client."""
    if limit <= 0:
        return
    ip = _client_key(request)
    now = time.monotonic()
    with _lock:
        window = _hits.get(ip)
        if window is None:
            if len(_hits) >= _MAX_CLIENTS:
                _evict_locked(now)
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


def acquire_inflight(key: str, max_inflight: int = MAX_INFLIGHT_PER_CLIENT) -> None:
    """Reserve one concurrent-request slot for `key`, or raise 429."""
    with _lock:
        current = _inflight.get(key, 0)
        if current >= max_inflight:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="too many concurrent requests",
                headers={"Retry-After": "1"},
            )
        _inflight[key] = current + 1


def release_inflight(key: str) -> None:
    with _lock:
        current = _inflight.get(key, 0) - 1
        if current > 0:
            _inflight[key] = current
        else:
            _inflight.pop(key, None)


def query_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(_settings_dep)],
) -> Iterator[None]:
    check_rate_limit(request, int(getattr(settings, "QUERY_RATE_LIMIT_PER_MINUTE", 0)))
    key = _client_key(request)
    acquire_inflight(key)
    try:
        yield
    finally:
        release_inflight(key)


RateLimited = Annotated[None, Depends(query_rate_limit)]
