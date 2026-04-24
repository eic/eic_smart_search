import re
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup


@dataclass(slots=True)
class ExtractedHtml:
    title: str
    content: str
    headings: list[str] = field(default_factory=list)
    breadcrumbs: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    last_updated: datetime | None = None


NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "header",
    "footer",
    ".site-search-panel",
    ".site-navbar",
    ".breadcrumb",
    ".breadcrumbs",
    ".toc",
    ".table-of-contents",
]


def extract_html(html: str, url: str, last_modified_header: str | None = None) -> ExtractedHtml:
    soup = BeautifulSoup(html, "html.parser")
    title = _title(soup, url)
    breadcrumbs = _breadcrumbs(soup)
    links = [
        urljoin(url, href)
        for href in [node.get("href") for node in soup.find_all("a")]
        if href and not href.startswith(("#", "mailto:", "tel:", "javascript:"))
    ]
    for selector in NOISE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup.body or soup
    headings: list[str] = []
    blocks: list[str] = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "table"]):
        text = _clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name and node.name.startswith("h"):
            level = int(node.name[1])
            headings.append(text)
            blocks.append(f"{'#' * level} {text}")
        elif node.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)
    content = "\n\n".join(_dedupe_adjacent(blocks))
    return ExtractedHtml(
        title=title,
        content=content,
        headings=headings,
        breadcrumbs=breadcrumbs,
        links=links,
        last_updated=_parse_http_date(last_modified_header),
    )


def _title(soup: BeautifulSoup, url: str) -> str:
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return _clean_text(og_title["content"])
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return _clean_text(h1.get_text(" ", strip=True))
    return url.rstrip("/").split("/")[-1] or url


def _breadcrumbs(soup: BeautifulSoup) -> list[str]:
    candidates = soup.select("[aria-label='breadcrumb'] li, .breadcrumb li, .breadcrumbs li, .breadcrumb a, .breadcrumbs a")
    return [_clean_text(node.get_text(" ", strip=True)) for node in candidates if _clean_text(node.get_text(" ", strip=True))]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_adjacent(blocks: list[str]) -> list[str]:
    output: list[str] = []
    previous = ""
    for block in blocks:
        if block == previous:
            continue
        output.append(block)
        previous = block
    return output


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

