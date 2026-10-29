"""BLEU / corpus metrics."""
from __future__ import annotations


def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    """Return corpus BLEU (0–100). Prefer sacrebleu; fall back to simple 1-gram."""
    try:
        import sacrebleu

        # Chinese: tokenize with zh; English: 13a. Mix → use sacrebleu default 13a
        # and let user pass --tokenize when evaluating a single direction.
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        return float(bleu.score)
    except Exception:
        # Tiny fallback: character / word overlap proxy (not a real BLEU).
        if not hyps:
            return 0.0
        hits = 0
        total = 0
        for h, r in zip(hyps, refs):
            ht = set(h)
            rt = set(r)
            hits += len(ht & rt)
            total += max(1, len(rt))
        return 100.0 * hits / total


def bleu_for_direction(hyps: list[str], refs: list[str], direction: str) -> float:
    try:
        import sacrebleu

        tok = "zh" if direction == "en2zh" else "13a"
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
        return float(bleu.score)
    except Exception:
        return corpus_bleu(hyps, refs)
