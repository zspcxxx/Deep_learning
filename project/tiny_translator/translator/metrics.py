"""BLEU over character tokens (the usual choice for Chinese output)."""
from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps: list[list[str]], refs: list[list[str]], max_n: int = 4) -> dict:
    """Corpus BLEU with one reference per sentence; zero n-gram counts are smoothed (Chen & Cherry method 1)."""
    matches = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            h, r = _ngrams(hyp, n), _ngrams(ref, n)
            matches[n - 1] += sum(min(c, r[g]) for g, c in h.items())
            totals[n - 1] += max(0, len(hyp) - n + 1)
    precisions = []
    for m, t in zip(matches, totals):
        if t == 0:
            precisions.append(0.0)
        else:
            precisions.append((m if m else 0.1) / t)
    if hyp_len == 0 or min(precisions) == 0:
        return {"bleu": 0.0, "precisions": [round(100 * p, 2) for p in precisions], "bp": 0.0,
                "hyp_len": hyp_len, "ref_len": ref_len}
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return {"bleu": round(100 * bleu, 2), "precisions": [round(100 * p, 2) for p in precisions],
            "bp": round(bp, 4), "hyp_len": hyp_len, "ref_len": ref_len}


def sentence_bleu(hyp: list[str], ref: list[str]) -> float:
    return corpus_bleu([hyp], [ref])["bleu"]
