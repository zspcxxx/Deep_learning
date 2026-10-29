#!/usr/bin/env python3
"""Build project/mt_data/kids-dialogue.tsv for children's conversational EN↔ZH MT.

Sources:
  1) Hand-curated kids/family dialogue seed (resources/kids_seed.tsv)
  2) Tatoeba cmn-eng short pairs filtered for kid-domain oral language

Run from anywhere:
  PYTHONPATH=project/bilingual_mt python project/bilingual_mt/scripts/build_kids_dialogue.py
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MT_DATA = ROOT.parent / "mt_data"
SEED = ROOT / "resources" / "kids_seed.tsv"
CMN_URL = "https://www.manythings.org/anki/cmn-eng.zip"

KID = re.compile(
    r"\b(mom|mum|mommy|mummy|dad|daddy|grandma|grandpa|brother|sister|friend|friends|"
    r"school|teacher|play|playing|game|toy|toys|hungry|thirsty|sleep|sleepy|bed|eat|"
    r"drink|please|thank|thanks|sorry|love|happy|sad|scared|tired|dog|cat|bird|fish|"
    r"book|draw|sing|park|ball|home|water|apple|banana|milk|hello|hi|bye|birthday|"
    r"hug|story|song|fun|cute|yummy|baby|color|colour|red|blue|green|yellow|pink|"
    r"cookie|cake|juice|candy|swing|slide|cartoon|homework|pencil|bag|shoes|wash|"
    r"brush|teeth|hands|breakfast|dinner|lunch|noodles|rice|ice cream)\b",
    re.I,
)
ORAL = re.compile(
    r"\b(i|i'm|you|you're|we|we're|my|your|let's|can i|do you|are you|what|where|who|how|why)\b|\?",
    re.I,
)
BAD = re.compile(
    r"\b(kill|dead|death|die|war|sex|drug|alcohol|beer|wine|cigarette|politic|president|"
    r"murder|suicide|government|prison|gun|bomb|terror|hell|damn|fuck|shit|professor|"
    r"publication|article|customer|customers|business|company|office|salary|invest|"
    r"stock|court|lawyer|police|arrest|tax|election|military|doctor|matter|agree|"
    r"serious|sell|troubled)\b",
    re.I,
)
ZH_BAD = re.compile(r"死|杀|战争|政治|总统|烟|酒|性|毒品|枪|爆炸|教授|公司|工资|警察|法院|投资|顾客|文章")


def normalize_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("\u00a0", " ").strip()
    return re.sub(r"\s+", " ", s)


def normalize_zh(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("\u00a0", " ").strip()
    return re.sub(r"\s+", "", s)


def ok_pair(en: str, zh: str) -> bool:
    words = re.findall(r"[A-Za-z']+", en)
    if not (1 <= len(words) <= 8):
        return False
    if not (1 <= len(zh) <= 16):
        return False
    if BAD.search(en) or ZH_BAD.search(zh):
        return False
    if '"' in en:
        return False
    if not ORAL.search(en):
        return False
    if not KID.search(en):
        return False
    return True


def load_seed(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((normalize_en(parts[0]), normalize_zh(parts[1])))
    return rows


def parse_anki_zip(zip_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(name) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def ensure_cmn_eng(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = raw_dir / "cmn-eng.zip"
    if zpath.is_file() and zpath.stat().st_size > 1000:
        return zpath
    # Prefer already-shared cmn-eng.tsv rebuild path via re-download of zip.
    sys.path.insert(0, str(ROOT))
    from mt.data import _download  # noqa: WPS433

    _download(CMN_URL, zpath)
    return zpath


def build(out: Path, force_download: bool = False) -> Path:
    if not SEED.is_file():
        raise FileNotFoundError(f"缺少种子语料 {SEED}")
    seed = load_seed(SEED)
    raw_dir = MT_DATA / "_raw_kids"
    if force_download:
        z = raw_dir / "cmn-eng.zip"
        if z.is_file():
            z.unlink()
    zpath = ensure_cmn_eng(raw_dir)
    filtered: list[tuple[str, str]] = []
    for en, zh in parse_anki_zip(zpath):
        en, zh = normalize_en(en), normalize_zh(zh)
        if ok_pair(en, zh):
            filtered.append((en, zh))

    uniq: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for en, zh in seed + filtered:
        if not en or not zh:
            continue
        key = (en.lower(), zh)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((en, zh))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for en, zh in uniq:
            w.writerow([en, zh])
    print(f"写入 {out} ：{len(uniq)} 句（种子 {len(seed)} + 过滤 Tatoeba {len(filtered)}）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, default=MT_DATA / "kids-dialogue.tsv")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()
    build(args.output, force_download=args.force_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
