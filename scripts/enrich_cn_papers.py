#!/usr/bin/env python3
"""Extract figures from the English PDFs and patch the Chinese HTML guides.

- Locate every "Figure N:" / "SuppFigure N:" caption, gather the vector drawings and
  images above it (plus their axis labels and sub-captions), and render that region.
- Inject MathJax so the LaTeX formulas in the guides render.
- Insert <img> tags into empty <figure> / .figure-placeholder blocks.

Extraction needs PyMuPDF (``pip install pymupdf``); patching needs only the standard library.

    python scripts/enrich_cn_papers.py            # extract + patch
    python scripts/enrich_cn_papers.py --patch    # only rewrite HTML
    python scripts/enrich_cn_papers.py --extract  # only extract figures
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN_DIR = ROOT / "docs" / "papers" / "en"
CN_DIR = ROOT / "docs" / "papers" / "cn"
FIG_DIR = CN_DIR / "figures"

DPI = 170

MATHJAX = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
    processEscapes: true
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

FIG_CSS = """
  figure { text-align: center; margin: 1.6em auto; }
  figure img, .paper-fig { max-width: 100%; height: auto; border: 1px solid #e5e7eb; background: #fff; }
  figcaption, .caption { font-size: 0.95em; color: #444; margin-top: 0.5em; text-align: left; }
  .figure-placeholder { border: none; padding: 0; background: transparent; }
"""

# A caption must start its own text block and be followed by punctuation or a capitalised
# sentence, so in-text references such as "Figure 3 shows" are never taken as captions.
CAPTION_RE = re.compile(r"^(Supp)?(?:Figure|Fig\.)\s*(\d+(?:\.\d+)?)(?:\s*[:.]|\s+(?=[A-Z]))")
ANY_CAPTION_RE = re.compile(r"^(?:Supp)?(?:Figure|Fig\.|Table)\s*\d+(?:\.\d+)?\s*[:.]", re.I)
CN_SUPP_RE = re.compile(r"SuppFigure\s*(\d+)", re.I)
CN_FIG_RE = re.compile(r"图\s*(\d+(?:\.\d+)?)")
IMG_RE = re.compile(r'<img class="paper-fig" src="(figures/[^"]+)"[^>]*>\n?')

GROW_GAP = 22.0      # max vertical gap (pt) between pieces of one figure
SEED_GAP = 90.0      # max gap between the caption and the lowest piece of its figure
LABEL_PAD = 12.0     # text this close to the graphics is treated as a label


def pdf_ids() -> list[str]:
    return [pdf.stem.split("_", 1)[0] for pdf in sorted(EN_DIR.glob("*.pdf"))]


def pdf_for(arxiv: str) -> Path | None:
    matches = list(EN_DIR.glob(f"{arxiv}_*.pdf"))
    return matches[0] if matches else None


def html_for(arxiv: str) -> Path | None:
    matches = list(CN_DIR.glob(f"{arxiv}_*_cn.html"))
    return matches[0] if matches else None


def fig_stem(supp: bool, num: str) -> str:
    return ("supp" if supp else "fig") + num.replace(".", "_")


# ---------------------------------------------------------------- extraction

def _column(page_w: float, cap) -> tuple[float, float]:
    mid = page_w / 2
    center = (cap.x0 + cap.x1) / 2
    if cap.width > page_w * 0.5 or abs(center - mid) < page_w * 0.12:
        return 0.0, page_w
    return (0.0, mid + 4) if center < mid else (mid - 4, page_w)


def _h_overlap(r, lo: float, hi: float) -> float:
    return max(0.0, min(r.x1, hi) - max(r.x0, lo))


def _in_column(r, lo: float, hi: float) -> bool:
    return r.width > 0 and _h_overlap(r, lo, hi) >= 0.5 * r.width


HEADING_RE = re.compile(r"^(?:[A-Z]\.)?\d+(?:\.\d+)*\.?\s+[A-Z]")


def _is_paragraph(block, col_w: float) -> bool:
    rect, text = block
    return len(text) > 100 and rect.width > 0.55 * col_w


def _boxed(rect, graphics) -> bool:
    return any(g.width > 20 and g.height > 10 and (g + (-3, -3, 3, 3)).contains(rect) for g in graphics)


def _graphics(page):
    import pymupdf

    W, H = page.rect.width, page.rect.height
    area = W * H
    drawings = []
    for d in page.get_drawings():
        r = d["rect"]
        # Page backgrounds and full-page frames would swallow every cluster.
        if r.width * r.height > 0.5 * area:
            continue
        thin = min(r.width, r.height) < 2.5
        # Header/footer rules and footnote separators.
        if thin and (r.y1 < 60 or r.y0 > H - 60):
            continue
        drawings.append(d)
    rects = list(page.cluster_drawings(drawings=drawings)) if drawings else []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width * r.height <= 0.8 * area:
            rects.append(r)
    return [pymupdf.Rect(r) for r in rects if max(r.width, r.height) >= 6]


def figure_rect(page, cap, blocks, graphics, header: float = 0.0, merged=None):
    import pymupdf

    W = page.rect.width
    lo, hi = _column(W, cap)
    col_w = hi - lo

    def blocks_off(rect, text) -> bool:
        """Text that bounds a figure: body paragraphs, headings and other captions."""
        if rect == cap or not _in_column(rect, lo, hi) or _boxed(rect, graphics):
            return False
        return (_is_paragraph((rect, text), col_w) or bool(ANY_CAPTION_RE.match(text))
                or (len(text) < 80 and bool(HEADING_RE.match(text))))

    # Caption set beside the figure rather than below it.
    side = [
        g for g in graphics
        if g.height > cap.height and min(g.y1, cap.y1) - max(g.y0, cap.y0) > 0.5 * cap.height
        and (-10 <= cap.x0 - g.x1 < 30 or -10 <= g.x0 - cap.x1 < 30)
    ]
    if side:
        union = pymupdf.Rect(max(side, key=lambda g: g.width * g.height))
        for _ in range(3):
            probe = union + (-GROW_GAP / 2, -GROW_GAP / 2, GROW_GAP / 2, GROW_GAP / 2)
            for g in graphics:
                if g.intersects(probe) and not g.intersects(cap):
                    union |= g
            for rect, text in blocks:
                if rect != cap and len(text) < 100 and rect.intersects(probe) and not rect.intersects(cap):
                    union |= rect
        if cap.x0 >= union.x1 - 10:
            union.x1 = min(union.x1, cap.x0 - 1)
        else:
            union.x0 = max(union.x0, cap.x1 + 1)
        return (union + (-3, -3, 3, 3)) & page.rect

    # Drawings or image frames that run under the caption are cut off at the caption.
    graphics = [
        g if g.y1 <= cap.y0 + 3 else pymupdf.Rect(g.x0, g.y0, g.x1, cap.y0 - 1)
        for g in graphics
        if (g.y1 <= cap.y0 + 3 and _in_column(g, lo, hi)) or (g.y0 < cap.y0 - 15 and g.intersects(cap))
    ]
    # A frame or image reaching well into the other column is cut at the page centre.
    mid = W / 2
    if hi - lo < W:
        left = lo == 0
        graphics = [
            pymupdf.Rect(g.x0, g.y0, min(g.x1, mid), g.y1) if left and g.x1 > mid + 40
            else pymupdf.Rect(max(g.x0, mid), g.y0, g.x1, g.y1) if not left and g.x0 < mid - 40
            else g
            for g in graphics
        ]
        graphics = [g for g in graphics if not g.is_empty]

    above = [g for g in graphics if g.y1 <= cap.y0 + 3 and _in_column(g, lo, hi)]
    if above:
        seed = max(above, key=lambda g: g.y1)
        between = any(
            blocks_off(r, t) and r.y0 >= seed.y1 - 2 and r.y1 <= cap.y0 + 1 for r, t in blocks
        )
        if between or cap.y0 - seed.y1 > SEED_GAP:
            seed = None
    else:
        seed = None
    if seed is None:
        return _text_figure(page, cap, blocks, blocks_off, lo, hi, header)

    # Text ending beside the seed wraps around the figure; only text fully above it bounds it.
    barrier = max((r.y1 for r, t in blocks if r.y1 <= seed.y0 + 4 and blocks_off(r, t)), default=0.0)
    cands = [g for g in above if g.y0 >= barrier - 2]

    union = pymupdf.Rect(seed)
    used = {id(seed)}
    grown = True
    while grown:
        grown = False
        for g in cands:
            if id(g) in used:
                continue
            vgap = max(union.y0 - g.y1, g.y0 - union.y1, 0)
            if vgap <= GROW_GAP:
                union |= g
                used.add(id(g))
                grown = True

    # Axis labels, legends and sub-captions are text; pull in the ones touching the graphics.
    for _ in range(3):
        probe = pymupdf.Rect(union.x0 - LABEL_PAD, union.y0 - LABEL_PAD,
                             union.x1 + LABEL_PAD, cap.y0 - 0.5)
        changed = False
        for rect, text in blocks:
            if rect.y0 < barrier - 1 or rect.y1 > cap.y0 + 1 or rect == cap:
                continue
            if blocks_off(rect, text):
                continue
            # Axis labels hug the plot; text further out belongs to the other column.
            if not _in_column(rect, lo, hi) and not (
                rect.x0 >= union.x0 - LABEL_PAD - 30 and rect.x1 <= union.x1 + LABEL_PAD
            ):
                continue
            if not rect.intersects(probe) or rect in union:
                continue
            if rect.width > union.width + 2 * LABEL_PAD + 40:
                continue
            union |= rect
            changed = True
        if not changed:
            break

    # One image can hold several figures; stop at another caption or paragraph inside it.
    for rect, text in blocks:
        if rect.y1 < cap.y0 - 1 and rect.y0 > union.y0 and rect.intersects(union) and blocks_off(rect, text):
            union.y0 = max(union.y0, rect.y1 + 1)
    # Keep sub-figure labels that share the caption's text block.
    if merged is not None and merged.y0 < cap.y0 - 1:
        union |= pymupdf.Rect(merged.x0, merged.y0, merged.x1, cap.y0 - 1)
    return _finish(page, union, cap, header)


def _finish(page, union, cap, header: float = 0.0):
    import pymupdf

    top = header + 2 if union.y0 - 3 < header + 2 else union.y0 - 3
    union = pymupdf.Rect(union.x0 - 3, top, union.x1 + 3, min(union.y1 + 3, cap.y0 - 0.5))
    union &= page.rect
    if union.width < 30 or union.height < 20:
        return None
    return union


def _text_figure(page, cap, blocks, blocks_off, lo, hi, header: float = 0.0):
    """Figures drawn purely as text (e.g. attention visualizations): stack the label blocks."""
    import pymupdf

    labels = sorted(
        (r for r, t in blocks if r.y1 <= cap.y0 + 1 and _in_column(r, lo, hi) and r != cap),
        key=lambda r: -r.y1,
    )
    union = None
    for r in labels:
        if blocks_off(r, next(t for rr, t in blocks if rr == r)):
            break
        edge = cap.y0 if union is None else union.y0
        if edge - r.y1 > GROW_GAP:
            break
        union = pymupdf.Rect(r) if union is None else union | r
    if union is None or union.height < 60:
        return None
    return _finish(page, union, cap, header)


def trim_whitespace(path: Path, pad: int = 6) -> None:
    from PIL import Image, ImageChops

    im = Image.open(path).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255)))
    bbox = diff.point(lambda p: 0 if p < 16 else 255).getbbox()
    if not bbox:
        return
    l, t, r, b = bbox
    im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad))).save(path, optimize=True)


def extract_one(arxiv: str) -> int:
    import pymupdf

    pdf = pdf_for(arxiv)
    if pdf is None:
        return 0
    out_dir = FIG_DIR / arxiv
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    doc = pymupdf.open(pdf)
    pages = [
        [(pymupdf.Rect(b[:4]), b[4].strip()) for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        for page in doc
    ]

    # Running headers repeat on many pages; figures must start below them.
    def top_key(rect, text, page_h):
        if rect.y0 > 0.1 * page_h or len(text) > 120 or len(re.findall(r"[A-Za-z]", text)) < 8:
            return None
        return re.sub(r"\d+", "", text)

    counts: dict[str, int] = {}
    for page, blocks in zip(doc, pages):
        for key in {top_key(r, t, page.rect.height) for r, t in blocks} - {None}:
            counts[key] = counts.get(key, 0) + 1
    running = {k for k, n in counts.items() if n >= 3}

    saved: set[str] = set()
    for page, blocks in zip(doc, pages):
        header = max(
            (r.y1 for r, t in blocks if top_key(r, t, page.rect.height) in running), default=0.0
        )
        graphics = None
        for rect, text in blocks:
            merged = None
            m = CAPTION_RE.match(text)
            if not m:
                # Sub-figure labels such as "(c) Recall@10" can share the caption's block.
                lines = text.split("\n")
                idx = next((i for i, ln in enumerate(lines) if i and CAPTION_RE.match(ln.strip())), None)
                if idx is None or any(len(ln) > 40 for ln in lines[:idx]):
                    continue
                hit = lines[idx]
                m = CAPTION_RE.match(hit.strip())
                found = page.search_for(hit.strip()[:24], clip=rect)
                if not found:
                    continue
                merged = rect
                rect = pymupdf.Rect(rect.x0, found[0].y0, rect.x1, rect.y1)
            stem = fig_stem(bool(m.group(1)), m.group(2))
            if stem in saved:
                continue
            if graphics is None:
                graphics = _graphics(page)
            clip = figure_rect(page, rect, blocks, graphics, header, merged)
            if clip is None:
                print(f"  {arxiv} {stem}: no graphics above caption on p.{page.number + 1}")
                continue
            dest = out_dir / f"{stem}.png"
            page.get_pixmap(clip=clip, dpi=DPI).save(dest)
            trim_whitespace(dest)
            saved.add(stem)
            print(f"  {arxiv} {stem}: p.{page.number + 1} {clip.width:.0f}x{clip.height:.0f}pt")
    if not saved:
        out_dir.rmdir()
    return len(saved)


# ---------------------------------------------------------------- HTML patching

def inject_head(html: str) -> str:
    if "MathJax-script" not in html:
        html = re.sub(r"(<head[^>]*>)", r"\1\n" + MATHJAX, html, count=1, flags=re.I)
    if "figure img" not in html and ".paper-fig" not in html:
        html = re.sub(r"(</style>)", FIG_CSS + r"\1", html, count=1)
    return html


def drop_missing_images(html: str) -> str:
    return IMG_RE.sub(lambda m: m.group(0) if (CN_DIR / m.group(1)).is_file() else "", html)


def src_for(arxiv: str, caption: str) -> tuple[str, str] | None:
    supp = CN_SUPP_RE.search(caption)
    if supp:
        stem, label = fig_stem(True, supp.group(1)), f"SuppFigure {supp.group(1)}"
    else:
        num = CN_FIG_RE.search(caption)
        if not num:
            return None
        stem, label = fig_stem(False, num.group(1)), f"图 {num.group(1)}"
    if not (FIG_DIR / arxiv / f"{stem}.png").is_file():
        return None
    return f"figures/{arxiv}/{stem}.png", label


def _img(src: str, label: str) -> str:
    return f'<img class="paper-fig" src="{src}" alt="{label}">\n'


def patch_html(arxiv: str) -> None:
    html_path = html_for(arxiv)
    if html_path is None:
        return
    text = drop_missing_images(inject_head(html_path.read_text(encoding="utf-8")))
    inserted = 0

    def repl_figure(m: re.Match) -> str:
        nonlocal inserted
        block = m.group(0)
        if "<img" in block or "<svg" in block:
            return block
        cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", block, re.S | re.I)
        found = cap and src_for(arxiv, cap.group(1))
        if not found:
            return block
        inserted += 1
        return re.sub(r"(<figure[^>]*>)", lambda h: h.group(1) + "\n" + _img(*found), block, count=1, flags=re.I)

    def repl_placeholder(m: re.Match) -> str:
        nonlocal inserted
        block = m.group(0)
        if "<img" in block or "<svg" in block:
            return block
        found = src_for(arxiv, block)
        if not found:
            return block
        inserted += 1
        return re.sub(r'(<div class="figure-placeholder"[^>]*>)', lambda h: h.group(1) + "\n" + _img(*found),
                      block, count=1, flags=re.I)

    text = re.sub(r"<figure\b.*?</figure>", repl_figure, text, flags=re.S | re.I)
    text = re.sub(r'<div class="figure-placeholder"[^>]*>.*?</div>', repl_placeholder, text, flags=re.S | re.I)

    if "paper-fig" in text:
        text = re.sub(r"原文中的图[^。]*未嵌入图片，仅保留图注说明[。；]?", "原文图已按英文本裁切嵌入。", text)
        text = re.sub(r"原文中的图[^。]*仅保留图注[和及、]表格内容。", "原文图已按英文本裁切嵌入。", text)

    html_path.write_text(text, encoding="utf-8")
    print(f"  patched {html_path.name}: inserted {inserted} figures")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--only", nargs="*", help="arxiv ids to process")
    args = parser.parse_args()
    do_extract = args.extract or not args.patch
    do_patch = args.patch or not args.extract
    ids = args.only or pdf_ids()
    if do_extract:
        print("Extracting figures from English PDFs…")
        total = sum(extract_one(arxiv) for arxiv in ids)
        print(f"  {total} figures")
    if do_patch:
        print("Patching Chinese HTML…")
        for arxiv in ids:
            patch_html(arxiv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
