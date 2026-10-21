#!/usr/bin/env python3
"""Extract figures from English PDFs and patch Chinese HTML guides.

- Render each English PDF page and crop regions above "Figure N" captions.
- Inject MathJax so existing LaTeX formulas render.
- Insert <img> tags into empty <figure> / .figure-placeholder blocks.

    python scripts/enrich_cn_papers.py            # extract + patch
    python scripts/enrich_cn_papers.py --patch    # only rewrite HTML
    python scripts/enrich_cn_papers.py --extract  # only extract figures
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
EN_DIR = ROOT / "docs" / "papers" / "en"
CN_DIR = ROOT / "docs" / "papers" / "cn"
FIG_DIR = CN_DIR / "figures"

DPI = 160
SCALE = DPI / 72.0
PAD_PT = 6

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

CAPTION_RE = re.compile(
    r"^(?:Figure|Fig\.?|FIGURE)\s+(\d+(?:\.\d+)?)(?:\s*[:.\u2013\u2014-]|$)",
    re.I,
)
CN_FIG_RE = re.compile(r"图\s*(\d+(?:\.\d+)?)")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def pdf_ids() -> list[str]:
    ids = []
    for pdf in sorted(EN_DIR.glob("*.pdf")):
        ids.append(pdf.stem.split("_", 1)[0])
    return ids


def pdf_for(arxiv: str) -> Path | None:
    matches = list(EN_DIR.glob(f"{arxiv}_*.pdf"))
    return matches[0] if matches else None


def html_for(arxiv: str) -> Path | None:
    matches = list(CN_DIR.glob(f"{arxiv}_*_cn.html"))
    return matches[0] if matches else None


WORD_RE = re.compile(
    r'<word xMin="([^"]+)" yMin="([^"]+)" xMax="([^"]+)" yMax="([^"]+)">([^<]*)</word>'
)
PAGE_RE = re.compile(r'<page width="([^"]+)" height="([^"]+)">(.*?)</page>', re.S)
BLOCK_RE = re.compile(
    r'<block xMin="([^"]+)" yMin="([^"]+)" xMax="([^"]+)" yMax="([^"]+)">(.*?)</block>',
    re.S,
)
FIG_WORD_RE = re.compile(r"^(?:Figure|Fig\.?|FIGURE)$", re.I)
FIG_NUM_RE = re.compile(r"^(\d+(?:\.\d+)?)[:.\u2013\u2014-]?$")
CAPTION_NEXT_RE = re.compile(
    r"^(?:[A-Z(]|Left|Right|Top|Bottom|We|The|A |An |Inception|Residual|Training|Example|Language|As |When)"
)


def parse_bbox(pdf: Path) -> list[dict]:
    raw = subprocess.check_output(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        text=True,
        errors="replace",
    )
    pages = []
    for i, page_m in enumerate(PAGE_RE.finditer(raw), start=1):
        width = float(page_m.group(1))
        height = float(page_m.group(2))
        body = page_m.group(3)
        blocks = []
        words = []
        for block_m in BLOCK_RE.finditer(body):
            parts = []
            bx0, by0, bx1, by1 = map(float, block_m.group(1, 2, 3, 4))
            for wm in WORD_RE.finditer(block_m.group(5)):
                wx0, wy0, wx1, wy1 = map(float, wm.group(1, 2, 3, 4))
                token = wm.group(5)
                word = {
                    "text": token,
                    "xmin": wx0, "ymin": wy0, "xmax": wx1, "ymax": wy1,
                }
                words.append(word)
                parts.append(token)
            text = " ".join(p for p in parts if p).strip()
            if text:
                blocks.append({
                    "text": text,
                    "xmin": bx0, "ymin": by0, "xmax": bx1, "ymax": by1,
                })
        pages.append({
            "num": i, "width": width, "height": height,
            "blocks": blocks, "words": words,
        })
    return pages


def find_captions(pages: list[dict]) -> list[dict]:
    caps = []
    seen = set()
    for page in pages:
        words = page["words"]
        for i, word in enumerate(words[:-1]):
            if not FIG_WORD_RE.match(word["text"]):
                continue
            nxt = words[i + 1]
            raw_num = nxt["text"].strip("()[],;")
            num_m = FIG_NUM_RE.match(raw_num)
            if not num_m:
                continue
            # Skip in-text references ("see Figure 2") unless the number token
            # itself carries caption punctuation, or the following word looks
            # like the start of a caption sentence.
            has_punct = bool(re.search(r"[:.\u2013\u2014-]", nxt["text"]))
            nxt2 = words[i + 2]["text"] if i + 2 < len(words) else ""
            if not has_punct and not CAPTION_NEXT_RE.match(nxt2):
                continue
            key = (page["num"], num_m.group(1))
            if key in seen:
                continue
            seen.add(key)
            # Caption line: from "Figure" through nearby words on the same baseline.
            line = [word, nxt]
            for extra in words[i + 2 : i + 18]:
                if abs(extra["ymin"] - word["ymin"]) > 8:
                    break
                line.append(extra)
            caps.append({
                "page": page["num"],
                "num": num_m.group(1),
                "page_w": page["width"],
                "page_h": page["height"],
                "xmin": min(w["xmin"] for w in line),
                "ymin": min(w["ymin"] for w in line),
                "xmax": max(w["xmax"] for w in line),
                "ymax": max(w["ymax"] for w in line),
                "blocks": page["blocks"],
                "words": words,
            })
    return caps


def _is_header_or_footer(block: dict, page_h: float) -> bool:
    text = block["text"].strip()
    if block["ymax"] < 36 or block["ymin"] > page_h - 36:
        if len(text) < 40:
            return True
    if re.fullmatch(r"\d+", text):
        return True
    return False


def _pt_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    l, t, r, b = box
    return (
        max(0, int(l * SCALE)),
        max(0, int(t * SCALE)),
        min(w, int(r * SCALE)),
        min(h, int(b * SCALE)),
    )


def _ink_bbox(ink: np.ndarray, pad: int = 6) -> tuple[int, int, int, int] | None:
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    if not rows.any() or not cols.any():
        return None
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    t, b = int(ys[0]), int(ys[-1]) + 1
    l, r = int(xs[0]), int(xs[-1]) + 1
    return (
        max(0, l - pad),
        max(0, t - pad),
        min(ink.shape[1], r + pad),
        min(ink.shape[0], b + pad),
    )


def mask_text(ink: np.ndarray, cap: dict, extra_skip: list[dict] | None = None) -> np.ndarray:
    out = ink.copy()
    skip = extra_skip or []
    h, w = out.shape
    for block in cap["blocks"]:
        if _is_header_or_footer(block, cap["page_h"]):
            continue
        if len(block["text"]) < 48:
            continue
        # Keep caption pixels unmasked so we can still find nearby ink; they
        # sit below the figure and are trimmed later.
        if abs(block["ymin"] - cap["ymin"]) < 10 and abs(block["xmin"] - cap["xmin"]) < 40:
            continue
        x0, y0, x1, y1 = _pt_box((block["xmin"] - 1, block["ymin"] - 1, block["xmax"] + 1, block["ymax"] + 1), w, h)
        out[y0:y1, x0:x1] = False
    for block in skip:
        x0, y0, x1, y1 = _pt_box((block["xmin"], block["ymin"], block["xmax"], block["ymax"]), w, h)
        out[y0:y1, x0:x1] = False
    return out


def search_regions(cap: dict) -> list[tuple[float, float, float, float]]:
    page_w, page_h = cap["page_w"], cap["page_h"]
    mid = page_w / 2
    others = [
        b for b in cap["blocks"]
        if not _is_header_or_footer(b, page_h)
        and abs(b["ymin"] - cap["ymin"]) > 6
    ]
    above = [b for b in others if b["ymax"] < cap["ymin"] - 3]
    top = max((b["ymax"] for b in above), default=42) + 2
    if cap["ymin"] - top < 36:
        top = max(36.0, cap["ymin"] - min(320.0, page_h * 0.45))
    bottom = cap["ymin"] - 1
    center = (cap["xmin"] + cap["xmax"]) / 2
    full = (cap["xmax"] - cap["xmin"]) > page_w * 0.45 or abs(center - mid) < page_w * 0.16
    regions = []
    if full:
        regions.append((28.0, top, page_w - 28.0, bottom))
    else:
        if center < mid:
            regions.append((28.0, top, mid - 4, bottom))
            regions.append((mid + 4, max(36.0, cap["ymin"] - 260), page_w - 28.0, cap["ymax"] + 8))
        else:
            regions.append((mid + 4, top, page_w - 28.0, bottom))
            regions.append((28.0, max(36.0, cap["ymin"] - 260), mid - 4, cap["ymax"] + 8))
    # Last resort: a wide band above the caption.
    regions.append((28.0, max(36.0, cap["ymin"] - 300), page_w - 28.0, bottom))
    return regions


def crop_figure_image(im: Image.Image, cap: dict) -> Image.Image | None:
    gray = np.array(im.convert("L"))
    ink = gray < 232
    masked = mask_text(ink, cap)
    best = None
    best_area = 0
    for region in search_regions(cap):
        x0, y0, x1, y1 = _pt_box(region, im.width, im.height)
        if x1 - x0 < 30 or y1 - y0 < 30:
            continue
        sub = masked[y0:y1, x0:x1]
        box = _ink_bbox(sub)
        if box is None:
            continue
        l, t, r, b = box
        area = (r - l) * (b - t)
        density = sub[t:b, l:r].mean() if area else 0
        if area < 50 * 50 or density < 0.0015:
            continue
        if area > best_area:
            best_area = area
            best = (x0 + l, y0 + t, x0 + r, y0 + b)
    if best is None:
        return None
    return trim_whitespace(im.crop(best))


def trim_whitespace(im: Image.Image, pad: int = 8) -> Image.Image:
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    # Treat near-white as background.
    diff = diff.point(lambda p: 0 if p < 18 else p)
    bbox = diff.getbbox()
    if not bbox:
        return im
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(im.width, r + pad)
    b = min(im.height, b + pad)
    return im.crop((l, t, r, b))


def render_page(pdf: Path, page: int, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = dest_dir / f"page-{page}"
    run([
        "pdftocairo", "-png", "-r", str(DPI),
        "-f", str(page), "-l", str(page),
        str(pdf), str(prefix),
    ])
    # pdftocairo appends a page number: page-3-3.png or page-3.png
    matches = sorted(dest_dir.glob(f"{prefix.name}*.png"))
    if not matches:
        raise FileNotFoundError(f"no render for {pdf} page {page}")
    return matches[0]


def extract_one(arxiv: str) -> list[Path]:
    pdf = pdf_for(arxiv)
    if pdf is None:
        return []
    out_dir = FIG_DIR / arxiv
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = parse_bbox(pdf)
    caps = find_captions(pages)
    written: list[Path] = []
    if not caps:
        print(f"  {arxiv}: no Figure captions found")
        return written
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rendered: dict[int, Path] = {}
        for cap in caps:
            page = cap["page"]
            if page not in rendered:
                rendered[page] = render_page(pdf, page, tmp_path)
            im = Image.open(rendered[page]).convert("RGB")
            crop = crop_figure_image(im, cap)
            if crop is None or crop.size[0] < 40 or crop.size[1] < 40:
                print(f"  {arxiv} fig {cap['num']}: no ink region, skip")
                continue
            dest = out_dir / f"fig{cap['num'].replace('.', '_')}.png"
            if dest.exists():
                print(f"  {arxiv} fig {cap['num']}: already saved, keep first crop")
                continue
            crop.save(dest, optimize=True)
            written.append(dest)
            print(f"  {arxiv} fig {cap['num']}: {crop.size[0]}x{crop.size[1]} -> {dest.name}")
    return written


def inject_head(html: str) -> str:
    if "MathJax-script" not in html:
        html = re.sub(r"(<head[^>]*>)", r"\1\n" + MATHJAX, html, count=1, flags=re.I)
    if "figure img" not in html and ".paper-fig" not in html:
        html = re.sub(r"(</style>)", FIG_CSS + r"\1", html, count=1)
    return html


def fig_src(arxiv: str, num: str) -> str | None:
    path = FIG_DIR / arxiv / f"fig{num.replace('.', '_')}.png"
    if path.is_file():
        return f"figures/{arxiv}/{path.name}"
    return None


def insert_into_figure(block: str, src: str, num: str) -> str:
    if "<img" in block or "<svg" in block:
        return block
    img = f'<img class="paper-fig" src="{src}" alt="图 {num}">\n'
    if "<figcaption" in block:
        return re.sub(r"(<figure[^>]*>)", r"\1\n" + img, block, count=1, flags=re.I)
    return block


def insert_into_placeholder(block: str, src: str, num: str) -> str:
    if "<img" in block or "<svg" in block:
        return block
    img = f'<img class="paper-fig" src="{src}" alt="图 {num}">\n'
    return re.sub(
        r'(<div class="figure-placeholder"[^>]*>)',
        r"\1\n" + img,
        block,
        count=1,
        flags=re.I,
    )


def patch_html(arxiv: str) -> None:
    html_path = html_for(arxiv)
    if html_path is None:
        return
    text = html_path.read_text(encoding="utf-8")
    text = inject_head(text)
    inserted = 0

    def repl_figure(m: re.Match) -> str:
        nonlocal inserted
        block = m.group(0)
        cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", block, re.S | re.I)
        if not cap:
            return block
        num_m = CN_FIG_RE.search(cap.group(1))
        if not num_m:
            return block
        src = fig_src(arxiv, num_m.group(1))
        if not src:
            return block
        out = insert_into_figure(block, src, num_m.group(1))
        if out != block:
            inserted += 1
        return out

    text = re.sub(r"<figure\b.*?</figure>", repl_figure, text, flags=re.S | re.I)

    def repl_ph(m: re.Match) -> str:
        nonlocal inserted
        block = m.group(0)
        num_m = CN_FIG_RE.search(block)
        if not num_m:
            return block
        src = fig_src(arxiv, num_m.group(1))
        if not src:
            return block
        out = insert_into_placeholder(block, src, num_m.group(1))
        if out != block:
            inserted += 1
        return out

    text = re.sub(r'<div class="figure-placeholder"[^>]*>.*?</div>', repl_ph, text, flags=re.S | re.I)

    # Drop the "figures were not embedded" disclaimer once images exist.
    if inserted:
        text = re.sub(
            r"原文中的图[^。]*未嵌入图片，仅保留图注说明[。；]?",
            "原文图已按英文本裁切嵌入。",
            text,
        )
        text = re.sub(
            r"原文中的图[^。]*仅保留图注[和及、]表格内容。",
            "原文图已按英文本裁切嵌入。",
            text,
        )

    html_path.write_text(text, encoding="utf-8")
    print(f"  patched {html_path.name}: +MathJax, inserted {inserted} figures")


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
        for arxiv in ids:
            extract_one(arxiv)
    if do_patch:
        print("Patching Chinese HTML…")
        for arxiv in ids:
            patch_html(arxiv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
