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


def extract_html(
    html: str,
    url: str,
    last_modified_header: str | None = None,
    prefer_trafilatura: bool = True,
) -> ExtractedHtml:
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
    # trafilatura is markedly better at boilerplate removal on generic
    # pages. Its markdown output keeps `#` headings, which the
    # HeadingAwareChunker relies on. Feed it the already noise-stripped
    # soup so NOISE_SELECTORS removal applies on both paths. The bs4 walk
    # above stays as the fallback (and the headings/links/breadcrumbs
    # source of truth): trafilatura sometimes drops code-heavy tutorial
    # bodies, so only prefer it when it kept a comparable amount of content.
    if prefer_trafilatura:
        traf_content = _trafilatura_markdown(str(soup), url)
        if traf_content and len(traf_content) >= 0.5 * len(content):
            content = traf_content
    return ExtractedHtml(
        title=title,
        content=content,
        headings=headings,
        breadcrumbs=breadcrumbs,
        links=links,
        last_updated=_parse_http_date(last_modified_header),
    )


def _trafilatura_markdown(html: str, url: str) -> str:
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        out = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_tables=True,
            include_formatting=True,
            include_links=False,
            favor_recall=True,
        )
    except Exception:
        return ""
    return (out or "").strip()


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

