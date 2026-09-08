"""SSRF guard for ingest-time downloads of remote-supplied URLs.

Zenodo/Indico give us file URLs from their JSON; blindly fetching them (with
redirects) lets a crafted record probe the internal Docker network (Qdrant
has no API key) or a cloud metadata endpoint. Enforce an allowlist of public
hosts and reject private/loopback IPs on every hop.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

ALLOWED_HOSTS = {
    "zenodo.org",
    "indico.bnl.gov",
    "arxiv.org",
    "export.arxiv.org",
    "inspirehep.net",
    "api.github.com",
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
}


class BlockedURLError(Exception):
    pass


def _host_allowed(host: str) -> bool:
    host = (host or "").lower()
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def _resolves_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def check_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise BlockedURLError(f"scheme not allowed: {url[:80]}")
    host = parsed.hostname or ""
    if not _host_allowed(host):
        raise BlockedURLError(f"host not in ingest allowlist: {host}")
    if not _resolves_public(host):
        raise BlockedURLError(f"host resolves to a non-public address: {host}")


def safe_get(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    """GET with per-hop allowlist enforcement. Follows redirects manually so
    a 302 into the internal network is rejected, not chased."""
    check_url(url)
    kwargs.pop("follow_redirects", None)
    current = url
    for _ in range(5):
        response = client.get(current, follow_redirects=False, **kwargs)
        if response.is_redirect and response.headers.get("location"):
            current = str(response.next_request.url) if response.next_request else response.headers["location"]
            check_url(current)
            continue
        return response
    raise BlockedURLError("too many redirects")
