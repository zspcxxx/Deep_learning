#!/usr/bin/env python3
"""Build child-friendly EN↔ZH corpora under project/mt_data/ from OPUS parallel data.

Each source is filtered for short, everyday, kid-appropriate spoken language and written
as its own TSV (English<Tab>Chinese, Chinese in simplified characters):

  kids-subtitles  OpenSubtitles v2024 en-zh_CN   movie & cartoon dialogue (22M lines → filtered)
  kids-tatoeba    Tatoeba 2026-07 cmn-en         volunteer-written everyday sentences
  kids-mix        hand-written seed (resources/kids_seed.tsv) + both of the above, de-duplicated

(QED educational subtitles were tried and dropped: mostly Cantonese and misaligned lines.)
Both translator projects call build(); it can also be run directly:

    python project/bilingual_mt/scripts/build_kids_corpora.py            # all three
    python project/bilingual_mt/scripts/build_kids_corpora.py tatoeba    # just one
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MT_DATA = ROOT.parent / "mt_data"
RAW = MT_DATA / "_raw_kids"
SEED = ROOT / "resources" / "kids_seed.tsv"
OPUS = "https://object.pouta.csc.fi/OPUS-"

SOURCES = {
    "subtitles": {
        "url": OPUS + "OpenSubtitles/v2024/moses/en-zh_CN.txt.zip",
        "en": ".en", "zh": ".zh_CN", "max_words": 10, "anchor": True,
    },
    "tatoeba": {
        "url": OPUS + "Tatoeba/v2026-07-08/moses/cmn-en.txt.zip",
        "en": ".en", "zh": ".cmn", "max_words": 10, "anchor": False,
    },
}
MAX_PER_ENGLISH = 2  # subtitles repeat "Thank you." thousands of times with varied wording

# ---------------------------------------------------------------- what counts as kids' talk

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
EN_BAD = re.compile(
    r"\b(kill\w*|dead|death|die|war|sex\w*|drug\w*|alcohol|beer|wine|cigarette|politic\w*|president|"
    r"murder|suicide|government|prison|gun|bomb|terror\w*|hell|damn\w*|professor|publication|"
    r"article|customers?|business|company|office|salary|invest\w*|stock|court|lawyer|police|"
    r"arrest|tax|election|military|doctor|matter|agree|serious|sell|troubled|bastard|crap|"
    r"asshole|ass|whore|slut|dick|piss|bloody|blood|drunk|naked|stupid|idiot|suck\w*)\b"
    r"|fuck|shit|bitch|shut up",
    re.I,
)
ZH_BAD = re.compile(
    r"死|杀|战争|政治|总统|烟|酒|性|毒品|枪|爆炸|教授|公司|工资|警察|法院|投资|顾客|文章|"
    r"他妈|妈的|操你|混蛋|王八|婊|贱|傻逼|狗屎|见鬼|滚|屁|裸|血|醉|白痴|蠢|笨蛋|闭嘴"
)
CANTONESE = re.compile(r"[佢嘅喺咁哋冇嘢啲咗嚟睇唔乜嗰噉晒囉喇咩]")

# Subtitle lines are often misaligned. Keep a pair only if a kid-topic word on the English
# side has one of its usual Chinese renderings on the Chinese side.
ANCHORS = {
    r"mom|mum|mommy|mummy": "妈|母", r"dad|daddy": "爸|父",
    r"grandma": "奶|婆|姥|祖母", r"grandpa": "爷|外公|姥爷|祖父",
    r"brother": "哥|弟|兄", r"sister": "姐|妹", r"friends?": "朋友|友",
    r"school": "学", r"teacher": "老师|教", r"play|playing": "玩|演|打|踢|弹|游戏|比赛",
    r"game": "游戏|比赛|玩", r"toys?": "玩具", r"hungry": "饿", r"thirsty": "渴",
    r"sleep|sleepy": "睡|困|觉", r"bed": "床|睡", r"eat": "吃", r"drink": "喝|饮",
    r"please": "请|拜托|麻烦|求", r"thanks?": "谢", r"sorry": "对不起|抱歉|不好意思|遗憾",
    r"love": "爱|喜欢", r"happy": "开心|高兴|快乐|幸福|乐", r"sad": "伤心|难过|悲",
    r"scared": "怕|吓", r"tired": "累|困|疲|厌", r"dog": "狗|犬", r"cat": "猫", r"bird": "鸟",
    r"fish": "鱼", r"book": "书", r"draw": "画", r"sing": "唱|歌", r"park": "公园",
    r"ball": "球", r"home": "家|回", r"water": "水", r"apple": "苹果", r"banana": "香蕉",
    r"milk": "奶", r"hello|hi": "你好|您好|嗨|喂|哈喽|好", r"bye": "再见|拜|走",
    r"birthday": "生日", r"hug": "抱", r"story": "故事", r"song": "歌|曲",
    r"fun": "好玩|有趣|乐|玩", r"cute": "可爱", r"yummy": "好吃|美味|香", r"baby": "宝|婴|孩|娃",
    r"colou?r": "色", r"red": "红", r"blue": "蓝", r"green": "绿", r"yellow": "黄", r"pink": "粉",
    r"cookie": "饼", r"cake": "糕", r"juice": "汁", r"candy": "糖", r"swing": "秋千|荡",
    r"slide": "滑", r"cartoon": "动画|卡通|漫画", r"homework": "作业|功课", r"pencil": "笔",
    r"bag": "包|袋", r"shoes": "鞋", r"wash": "洗", r"brush": "刷", r"teeth": "牙", r"hands": "手",
    r"breakfast": "早餐|早饭|早点", r"dinner": "晚餐|晚饭|饭|吃", r"lunch": "午餐|午饭|饭|吃",
    r"noodles": "面", r"rice": "米|饭", r"ice cream": "冰淇淋|冰激凌|雪糕",
}
ANCHOR_RULES = [(re.compile(rf"\b(?:{en})\b", re.I), re.compile(zh)) for en, zh in ANCHORS.items()]

CJK = re.compile(r"[\u4e00-\u9fff]")
NOISE = re.compile(r"[♪♫#\[\]()<>{}*_|~=@/\\\"]|https?:|www\.|字幕|翻译|校对|subtitle|caption", re.I)
LEAD_DASH = re.compile(r"^\s*[-–—]+\s*")
# Two speakers on one line ("Hello. -Um, I'm good"), cut-off lines ("How may I...").
SPEAKER_SPLIT = re.compile(r"(?:^|[\s.!?])-+\s*\S")
TRUNCATED = re.compile(r"\.\.\.|…")
WELL_FORMED = re.compile(r"^[A-Z0-9'].*[.!?]$")
# NFKC turns Chinese punctuation into ASCII; turn it back.
ZH_PUNCT = str.maketrans({",": "，", "?": "？", "!": "！", ":": "：", ";": "；"})
ZH_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")


def _simplify():
    try:
        import zhconv
    except ImportError:
        return lambda s: s
    return lambda s: zhconv.convert(s, "zh-cn")


def clean_en(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", LEAD_DASH.sub("", text)).strip()


def clean_zh(text: str, simplify) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    zh = simplify(re.sub(r"\s+", "", LEAD_DASH.sub("", text))).replace("...", "…")
    return ZH_PERIOD.sub("。", zh.translate(ZH_PUNCT))


def keep_en(en: str, max_words: int) -> int:
    """Word count if the English side qualifies, else 0. Cheap, so it runs first."""
    words = re.findall(r"[A-Za-z']+", en)
    if not (1 <= len(words) <= max_words) or not WELL_FORMED.match(en):
        return 0
    if NOISE.search(en) or EN_BAD.search(en) or SPEAKER_SPLIT.search(en) or TRUNCATED.search(en):
        return 0
    return len(words) if KID.search(en) and ORAL.search(en) else 0


def keep_zh(en: str, zh: str, n_words: int, max_words: int, anchor: bool) -> bool:
    if not (1 <= len(zh) <= 2 * max_words + 4):
        return False
    if NOISE.search(zh) or CANTONESE.search(zh) or TRUNCATED.search(zh) or ZH_BAD.search(zh):
        return False
    cjk = len(CJK.findall(zh))
    body = [c for c in zh if not unicodedata.category(c).startswith(("P", "S", "Z"))]
    if cjk == 0 or any(not (CJK.match(c) or c.isdigit()) for c in body):
        return False
    # Misaligned subtitle lines show up as wildly different lengths.
    if not (0.6 <= cjk / n_words <= 3.5):
        return False
    if anchor and not any(en_re.search(en) and zh_re.search(zh) for en_re, zh_re in ANCHOR_RULES):
        return False
    return True


# ---------------------------------------------------------------- download & build

def download(url: str, dest: Path, attempts: int = 3) -> Path:
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kids-corpora)"})
    for attempt in range(1, attempts + 1):
        print(f"下载 {url}（第 {attempt} 次）", flush=True)
        try:
            # The timeout applies to each socket read, so a stalled connection fails fast.
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
        except OSError as exc:
            print(f"下载失败：{exc}", flush=True)
            if attempt == attempts:
                raise
            continue
        tmp.replace(dest)
        return dest
    return dest


def stream_pairs(zip_path: Path, en_suffix: str, zh_suffix: str):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        en_name = next(n for n in names if n.endswith(en_suffix))
        zh_name = next(n for n in names if n.endswith(zh_suffix))
        with zf.open(en_name) as fe, zf.open(zh_name) as fz:
            yield from zip(io.TextIOWrapper(fe, encoding="utf-8", errors="replace"),
                           io.TextIOWrapper(fz, encoding="utf-8", errors="replace"))


def write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(rows)
    tmp.replace(path)


def read_tsv(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [(r[0], r[1]) for r in csv.reader(fh, delimiter="\t") if len(r) >= 2]


def build_source(name: str, keep_raw: bool = False) -> Path:
    src = SOURCES[name]
    zpath = download(src["url"], RAW / Path(src["url"]).name.replace(".txt.zip", f".{name}.zip"))
    simplify = _simplify()
    per_en: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str]] = []
    total = 0
    for en, zh in stream_pairs(zpath, src["en"], src["zh"]):
        total += 1
        en = clean_en(en)
        n_words = keep_en(en, src["max_words"])
        if not n_words:
            continue
        zh = clean_zh(zh, simplify)
        if not keep_zh(en, zh, n_words, src["max_words"], src["anchor"]):
            continue
        key_en = en.lower().rstrip(".!?")
        if (key_en, zh) in seen or per_en[key_en] >= MAX_PER_ENGLISH:
            continue
        seen.add((key_en, zh))
        per_en[key_en] += 1
        rows.append((en, zh))
    out = MT_DATA / f"kids-{name}.tsv"
    write_tsv(out, rows)
    print(f"kids-{name}: {total:,} 行 → 保留 {len(rows):,} 句（{len(per_en):,} 个不同英文句）→ {out}", flush=True)
    if not keep_raw:
        zpath.unlink(missing_ok=True)
    return out


def build_mix(keep_raw: bool = False) -> Path:
    simplify = _simplify()
    parts = {"seed": [(clean_en(en), clean_zh(zh, simplify)) for en, zh in read_tsv(SEED)]}
    for name in SOURCES:
        path = MT_DATA / f"kids-{name}.tsv"
        if not path.is_file():
            build_source(name, keep_raw)
        parts[name] = read_tsv(path)
    seen: set[tuple[str, str]] = set()
    rows = []
    for part in parts.values():
        for en, zh in part:
            key = (en.lower().rstrip(".!?"), zh)
            if key not in seen:
                seen.add(key)
                rows.append((en, zh))
    out = MT_DATA / "kids-mix.tsv"
    write_tsv(out, rows)
    counts = "，".join(f"{k} {len(v):,}" for k, v in parts.items())
    print(f"kids-mix: {len(rows):,} 句（{counts}，去重后）→ {out}", flush=True)
    return out


def build(dataset: str, keep_raw: bool = False) -> Path:
    """Build project/mt_data/<dataset>.tsv for dataset in kids-subtitles / kids-tatoeba / kids-mix."""
    name = dataset.removeprefix("kids-")
    if name == "mix":
        return build_mix(keep_raw)
    if name in SOURCES:
        return build_source(name, keep_raw)
    raise ValueError(f"未知儿童语料 {dataset!r}；可选 {DATASETS}")


DATASETS = [f"kids-{n}" for n in SOURCES] + ["kids-mix"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", choices=[*SOURCES, "mix"], help="默认全部")
    ap.add_argument("--keep-raw", action="store_true", help="保留下载的 OPUS 压缩包（默认用完即删）")
    args = ap.parse_args()
    for name in args.sources or [*SOURCES, "mix"]:
        build(f"kids-{name}", args.keep_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
