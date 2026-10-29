"""Text normalisation for EN / ZH parallel data."""
from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")


def normalize_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00a0", " ").strip()
    return _SPACE_RE.sub(" ", s)


def normalize_zh(s: str, simplified: bool = True) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00a0", " ").strip()
    s = _SPACE_RE.sub("", s)  # Chinese typically has no spaces
    if simplified:
        try:
            import zhconv

            s = zhconv.convert(s, "zh-cn")
        except Exception:
            pass
    return s


def clean_pair(en: str, zh: str, *, simplified: bool = True) -> tuple[str, str] | None:
    en = normalize_en(en)
    zh = normalize_zh(zh, simplified=simplified)
    if not en or not zh:
        return None
    if len(en) > 500 or len(zh) > 300:
        return None
    return en, zh
