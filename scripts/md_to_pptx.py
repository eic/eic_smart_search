#!/usr/bin/env python3
"""Convert PRESENTATION.md (Marp-flavored markdown with Mermaid) to .pptx.

- Splits slides on `---` lines
- Renders ```mermaid blocks via Kroki -> PNG and embeds them
- Renders markdown tables as native PowerPoint tables
- Strips Marp frontmatter and image-only directives

Usage:
    python3 scripts/md_to_pptx.py PRESENTATION.md PRESENTATION.pptx
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import requests
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

MERMAID_INK_SVG_URL = "https://mermaid.ink/svg/{token}"
RASTER_WIDTH_PX = 2400  # target PNG width after rasterization
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.4)
TITLE_H = Inches(0.7)
BODY_TOP = MARGIN + TITLE_H + Inches(0.1)
BODY_W = SLIDE_W - 2 * MARGIN
BODY_H = SLIDE_H - BODY_TOP - MARGIN

COLOR_TITLE = RGBColor(0x1F, 0x3A, 0x5F)
COLOR_BODY = RGBColor(0x22, 0x22, 0x22)
COLOR_ACCENT = RGBColor(0x2E, 0x86, 0xAB)
COLOR_TABLE_HEADER_BG = RGBColor(0x1F, 0x3A, 0x5F)
COLOR_TABLE_ALT_BG = RGBColor(0xF2, 0xF6, 0xFA)
COLOR_CODE_BG = RGBColor(0xF5, 0xF5, 0xF5)


def split_slides(md: str) -> list[str]:
    body = re.sub(r"^---\n.*?\n---\n", "", md, count=1, flags=re.DOTALL)
    parts = re.split(r"\n---\n", body)
    return [p.strip() for p in parts if p.strip()]


def render_mermaid(source: str, cache_dir: Path) -> tuple[bytes | None, bytes | None]:
    """Return (svg_bytes, png_fallback_bytes)."""
    cache_key = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    png_file = cache_dir / f"{cache_key}_hi.png"
    svg_file = cache_dir / f"{cache_key}.svg"
    svg_bytes = svg_file.read_bytes() if svg_file.exists() else None
    if svg_bytes is None:
        payload = json.dumps({"code": source, "mermaid": {"theme": "default"}}, separators=(",", ":"))
        token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
        url = MERMAID_INK_SVG_URL.format(token=token)
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=30)
                ct = r.headers.get("content-type", "")
                if r.status_code == 200 and "svg" in ct and r.content.lstrip().startswith(b"<svg"):
                    svg_bytes = r.content
                    svg_file.write_bytes(svg_bytes)
                    break
                print(f"  ! mermaid.ink {r.status_code} attempt {attempt+1} ({ct}, url len {len(url)})")
            except Exception as exc:
                print(f"  ! mermaid.ink error attempt {attempt+1}: {exc}")
            time.sleep(1.5)
    if svg_bytes is None:
        return None, None
    if png_file.exists():
        return svg_bytes, png_file.read_bytes()
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=RASTER_WIDTH_PX)
        png_file.write_bytes(png_bytes)
        return svg_bytes, png_bytes
    except Exception as exc:
        print(f"  ! cairosvg rasterize failed: {exc}")
        return svg_bytes, None


def svg_dimensions(svg_bytes: bytes) -> tuple[int, int]:
    """Best-effort width/height extraction from SVG root attrs / viewBox."""
    head = svg_bytes[:2048].decode("utf-8", errors="ignore")
    w = h = None
    m = re.search(r'\bwidth="([\d.]+)(?:px)?"', head)
    if m: w = float(m.group(1))
    m = re.search(r'\bheight="([\d.]+)(?:px)?"', head)
    if m: h = float(m.group(1))
    m = re.search(r'viewBox="[\d.\-\s]*?([\d.]+)\s+([\d.]+)"', head)
    vb_w = float(m.group(1)) if m else None
    vb_h = float(m.group(2)) if m else None
    if not w or w <= 1: w = vb_w
    if not h or h <= 1: h = vb_h
    return int(w or 1600), int(h or 900)


CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


def parse_blocks(body: str) -> list[dict]:
    """Split slide body into ordered blocks: heading, paragraph, list, table, mermaid, code."""
    blocks: list[dict] = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # Code fence
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            j = i + 1
            buf: list[str] = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            content = "\n".join(buf)
            if lang == "mermaid":
                blocks.append({"type": "mermaid", "content": content})
            else:
                blocks.append({"type": "code", "lang": lang, "content": content})
            i = j + 1
            continue
        # Heading
        if stripped.startswith("#"):
            m = re.match(r"(#{1,6})\s+(.*)", stripped)
            if m:
                blocks.append({"type": "heading", "level": len(m.group(1)), "content": m.group(2)})
                i += 1
                continue
        # Table
        if TABLE_LINE_RE.match(line) and i + 1 < len(lines) and TABLE_LINE_RE.match(lines[i + 1]) and "---" in lines[i + 1]:
            tbl_lines = [line]
            j = i + 1
            while j < len(lines) and TABLE_LINE_RE.match(lines[j]):
                tbl_lines.append(lines[j])
                j += 1
            blocks.append({"type": "table", "lines": tbl_lines})
            i = j
            continue
        # Blockquote
        if stripped.startswith(">"):
            buf = [stripped.lstrip("> ").rstrip()]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith(">"):
                buf.append(lines[j].strip().lstrip("> ").rstrip())
                j += 1
            blocks.append({"type": "quote", "content": "\n".join(buf)})
            i = j
            continue
        # List
        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            items: list[tuple[int, str]] = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i]) or (lines[i].strip() and lines[i].startswith(" "))):
                cur = lines[i]
                if not cur.strip():
                    break
                indent = len(cur) - len(cur.lstrip())
                text = re.sub(r"^\s*[-*]\s+", "", cur)
                text = re.sub(r"^\s*\d+\.\s+", "", text)
                items.append((indent // 2, text.strip()))
                i += 1
            blocks.append({"type": "list", "items": items})
            continue
        # Paragraph
        buf = [stripped]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(("#", "```", "-", "*", ">", "|")) and not re.match(r"^\s*\d+\.\s+", lines[j]):
            buf.append(lines[j].strip())
            j += 1
        blocks.append({"type": "para", "content": " ".join(buf)})
        i = j
    return blocks


def parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [[c.strip() for c in re.split(r"\s*\|\s*", ln.strip().strip("|"))] for ln in lines]
    header = rows[0]
    body = rows[2:] if len(rows) > 2 else []
    return header, body


INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
INLINE_ITALIC_RE = re.compile(r"(?<![*])\*(?!\*)([^*]+)\*(?!\*)")
INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def add_inline_runs(p, text: str, *, font_size=Pt(14), bold=False, color=COLOR_BODY) -> None:
    # Strip emoji presentation that could break some renderers? Keep them.
    # Tokenize: bold > code > italic > link > plain
    pattern = re.compile(
        r"(\*\*[^*]+\*\*)|(`[^`]+`)|(\[[^\]]+\]\([^)]+\))|(\*[^*]+\*)",
    )
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            _add_run(p, text[pos:m.start()], size=font_size, bold=bold, color=color)
        tok = m.group(0)
        if tok.startswith("**"):
            _add_run(p, tok[2:-2], size=font_size, bold=True, color=color)
        elif tok.startswith("`"):
            _add_run(p, tok[1:-1], size=font_size, bold=bold, color=color, mono=True)
        elif tok.startswith("["):
            lm = INLINE_LINK_RE.match(tok)
            if lm:
                _add_run(p, lm.group(1), size=font_size, bold=bold, color=COLOR_ACCENT, underline=True)
        elif tok.startswith("*"):
            _add_run(p, tok[1:-1], size=font_size, bold=bold, italic=True, color=color)
        pos = m.end()
    if pos < len(text):
        _add_run(p, text[pos:], size=font_size, bold=bold, color=color)


def _add_run(p, text, *, size, bold=False, italic=False, underline=False, color=COLOR_BODY, mono=False):
    run = p.add_run()
    run.text = text
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    run.font.color.rgb = color
    if mono:
        run.font.name = "Consolas"


def add_title(slide, title: str, subtitle: str | None) -> Emu:
    tb = slide.shapes.add_textbox(MARGIN, MARGIN, BODY_W, TITLE_H)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    add_inline_runs(p, title, font_size=Pt(28), bold=True, color=COLOR_TITLE)
    if subtitle:
        p2 = tf.add_paragraph()
        add_inline_runs(p2, subtitle, font_size=Pt(16), color=COLOR_ACCENT)
        return MARGIN + TITLE_H + Inches(0.25)
    return MARGIN + TITLE_H


def add_paragraph_box(slide, text: str, top: Emu, *, size=Pt(14)) -> Emu:
    height = Inches(max(0.4, 0.3 + 0.25 * (len(text) // 90 + 1)))
    tb = slide.shapes.add_textbox(MARGIN, top, BODY_W, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    add_inline_runs(p, text, font_size=size)
    return top + height + Inches(0.05)


def add_quote_box(slide, text: str, top: Emu) -> Emu:
    height = Inches(0.6 + 0.25 * (len(text) // 100))
    tb = slide.shapes.add_textbox(MARGIN, top, BODY_W, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.level = 0
    add_inline_runs(p, "▎ " + text, font_size=Pt(16), color=COLOR_ACCENT)
    return top + height + Inches(0.1)


def add_list_box(slide, items: list[tuple[int, str]], top: Emu, *, size=Pt(13)) -> Emu:
    line_h = 0.32
    height = Inches(0.2 + line_h * sum(max(1, len(t) // 90 + 1) for _, t in items))
    height = min(height, BODY_H - (top - BODY_TOP))
    tb = slide.shapes.add_textbox(MARGIN, top, BODY_W, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for idx, (level, text) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.level = min(level, 4)
        bullet = "• " if level == 0 else "  ▸ "
        add_inline_runs(p, bullet + text, font_size=size)
    return top + height + Inches(0.05)


def add_code_box(slide, code: str, top: Emu, *, size=Pt(11)) -> Emu:
    lines = code.count("\n") + 1
    height = Inches(0.25 + 0.22 * lines)
    height = min(height, BODY_H - (top - BODY_TOP))
    tb = slide.shapes.add_textbox(MARGIN, top, BODY_W, height)
    tb.fill.solid()
    tb.fill.fore_color.rgb = COLOR_CODE_BG
    tb.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD)
    tf = tb.text_frame
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    tf.word_wrap = False
    for idx, ln in enumerate(code.splitlines() or [""]):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = ln
        run.font.name = "Consolas"
        run.font.size = size
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return top + height + Inches(0.08)


def add_table_box(slide, header: list[str], rows: list[list[str]], top: Emu) -> Emu:
    n_rows = len(rows) + 1
    n_cols = len(header)
    row_h = Inches(0.3)
    height = row_h * n_rows
    if height > BODY_H - (top - BODY_TOP):
        height = BODY_H - (top - BODY_TOP)
    table_shape = slide.shapes.add_table(n_rows, n_cols, MARGIN, top, BODY_W, height)
    tbl = table_shape.table
    for c, h in enumerate(header):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_TABLE_HEADER_BG
        tf = cell.text_frame
        tf.margin_left = Inches(0.05)
        tf.margin_right = Inches(0.05)
        p = tf.paragraphs[0]
        p.clear()
        add_inline_runs(p, h, font_size=Pt(11), bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row[:n_cols]):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_TABLE_ALT_BG if r % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            tf = cell.text_frame
            tf.margin_left = Inches(0.05)
            tf.margin_right = Inches(0.05)
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.clear()
            add_inline_runs(p, val, font_size=Pt(10))
    return top + height + Inches(0.1)


SVG_EXT_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ASVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


def add_mermaid_box(slide, svg: bytes | None, png: bytes | None, top: Emu, remaining_h: Emu) -> Emu:
    if not svg and not png:
        return add_paragraph_box(slide, "[diagram render failed]", top)
    avail_w = BODY_W
    avail_h = remaining_h - Inches(0.1)
    if avail_h <= 0:
        avail_h = Inches(2)
    if svg:
        iw, ih = svg_dimensions(svg)
    else:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(png))
            iw, ih = img.size
        except Exception:
            iw, ih = 1600, 900
    aspect = iw / ih if ih else 1.0
    w = avail_w
    h = Emu(int(w / aspect))
    if h > avail_h:
        h = avail_h
        w = Emu(int(h * aspect))
    left = MARGIN + (BODY_W - w) // 2

    # Always add the PNG as the visible picture (universally rendered).
    pic = slide.shapes.add_picture(io.BytesIO(png), left, top, width=w, height=h) if png else None

    # If we also have SVG, splice it into the picture's blip so PowerPoint 2016+
    # and Google Slides render vector and keep editability/sharpness.
    if pic is not None and svg:
        _attach_svg_to_picture(slide, pic, svg)
    elif pic is None and svg:
        # No raster fallback — embed SVG only via the same path with a 1x1 PNG.
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGBA", (4, 4), (255, 255, 255, 0)).save(buf, format="PNG")
        buf.seek(0)
        pic = slide.shapes.add_picture(buf, left, top, width=w, height=h)
        _attach_svg_to_picture(slide, pic, svg)

    return top + h + Inches(0.1)


def _attach_svg_to_picture(slide, pic, svg_bytes: bytes) -> None:
    """Add an SVG blip extension to a picture so PowerPoint/Slides render vector."""
    from pptx.oxml.ns import qn
    from lxml import etree

    part = slide.part
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI
    package = part.package
    existing = {str(p.partname) for p in package.iter_parts()}
    n = 1
    while f"/ppt/media/image_mermaid_{n}.svg" in existing:
        n += 1
    partname = PackURI(f"/ppt/media/image_mermaid_{n}.svg")
    svg_part = Part(partname, "image/svg+xml", package, svg_bytes)
    rid = part.relate_to(svg_part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")

    blip = pic._element.find(".//" + qn("a:blip"))
    if blip is None:
        return
    # Append <a:extLst><a:ext uri="…"><asvg:svgBlip r:embed="rid"/></a:ext></a:extLst>
    nsmap = {"a": A_NS, "r": R_NS, "asvg": ASVG_NS}
    extLst = blip.find(qn("a:extLst"))
    if extLst is None:
        extLst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(extLst, qn("a:ext"))
    ext.set("uri", SVG_EXT_URI)
    svg_blip = etree.SubElement(ext, "{%s}svgBlip" % ASVG_NS)
    svg_blip.set("{%s}embed" % R_NS, rid)


def render_slide(prs: Presentation, raw: str, cache_dir: Path, idx: int, total: int) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    blocks = parse_blocks(raw)

    # Title + subtitle
    title_text = None
    subtitle_text = None
    rest_blocks = []
    for b in blocks:
        if title_text is None and b["type"] == "heading" and b["level"] <= 2:
            title_text = b["content"]
            continue
        if title_text is not None and subtitle_text is None and b["type"] == "heading" and b["level"] == 3:
            subtitle_text = b["content"]
            continue
        rest_blocks.append(b)

    if title_text is None:
        title_text = f"Slide {idx}"
    cursor = add_title(slide, title_text, subtitle_text)

    # Page number footer
    pn = slide.shapes.add_textbox(SLIDE_W - Inches(1.0), SLIDE_H - Inches(0.35), Inches(0.8), Inches(0.3))
    pn.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    _add_run(pn.text_frame.paragraphs[0], f"{idx} / {total}", size=Pt(9), color=RGBColor(0x99, 0x99, 0x99))

    slide_top_max = SLIDE_H - Inches(0.4)
    for b in rest_blocks:
        if cursor >= slide_top_max:
            break
        remaining = slide_top_max - cursor
        if b["type"] == "para":
            cursor = add_paragraph_box(slide, b["content"], cursor)
        elif b["type"] == "quote":
            cursor = add_quote_box(slide, b["content"], cursor)
        elif b["type"] == "list":
            cursor = add_list_box(slide, b["items"], cursor)
        elif b["type"] == "heading":
            cursor = add_paragraph_box(slide, "**" + b["content"] + "**", cursor, size=Pt(16))
        elif b["type"] == "code":
            cursor = add_code_box(slide, b["content"], cursor)
        elif b["type"] == "table":
            header, rows = parse_table(b["lines"])
            cursor = add_table_box(slide, header, rows, cursor)
        elif b["type"] == "mermaid":
            print(f"  · rendering mermaid ({len(b['content'])} chars)")
            svg, png = render_mermaid(b["content"], cache_dir)
            cursor = add_mermaid_box(slide, svg, png, cursor, remaining)


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("PRESENTATION.md")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("PRESENTATION.pptx")
    cache_dir = Path(".mermaid_cache")
    cache_dir.mkdir(exist_ok=True)

    md = src.read_text()
    slides = split_slides(md)
    print(f"Found {len(slides)} slides in {src}")

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    for i, raw in enumerate(slides, start=1):
        first_line = next((ln for ln in raw.split("\n") if ln.strip()), "")
        print(f"[{i}/{len(slides)}] {first_line[:70]}")
        render_slide(prs, raw, cache_dir, i, len(slides))

    prs.save(str(dst))
    print(f"\nWrote {dst} ({dst.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
