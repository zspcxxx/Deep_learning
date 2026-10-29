"""Joint SentencePiece BPE for English + Chinese."""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

# Special tokens (also used as SentencePiece user symbols).
PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"
# Direction tags prepended to the source so one model can do both directions.
TAG_EN2ZH = "<en2zh>"
TAG_ZH2EN = "<zh2en>"

SPECIALS = [PAD, UNK, BOS, EOS, TAG_EN2ZH, TAG_ZH2EN]


class Vocab:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))
        # Prefer explicit ids when present in the model.
        self.pad_id = self._id(PAD, fallback=self.sp.pad_id() if self.sp.pad_id() >= 0 else 0)
        self.unk_id = self._id(UNK, fallback=self.sp.unk_id())
        self.bos_id = self._id(BOS, fallback=self.sp.bos_id() if self.sp.bos_id() >= 0 else 1)
        self.eos_id = self._id(EOS, fallback=self.sp.eos_id() if self.sp.eos_id() >= 0 else 2)
        self.en2zh_id = self._id(TAG_EN2ZH)
        self.zh2en_id = self._id(TAG_ZH2EN)

    def _id(self, piece: str, fallback: int | None = None) -> int:
        i = self.sp.piece_to_id(piece)
        if i < 0 or (hasattr(self.sp, "unk_id") and i == self.sp.unk_id() and piece != UNK):
            if fallback is not None:
                return fallback
            raise KeyError(f"vocab 缺少特殊符号 {piece!r}")
        return i

    def __len__(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # Strip specials that SentencePiece might leave as pieces.
        clean = [
            i
            for i in ids
            if i not in (self.pad_id, self.bos_id, self.eos_id, self.en2zh_id, self.zh2en_id)
            and i >= 0
        ]
        return self.sp.decode(clean)

    def direction_tag_id(self, direction: str) -> int:
        if direction == "en2zh":
            return self.en2zh_id
        if direction == "zh2en":
            return self.zh2en_id
        raise ValueError(direction)

    def encode_source(self, text: str, direction: str, max_len: int) -> list[int]:
        """[dir_tag] + tokens + [eos], truncated to max_len."""
        body = self.encode(text)
        ids = [self.direction_tag_id(direction)] + body + [self.eos_id]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def encode_target(self, text: str, max_len: int) -> list[int]:
        """[bos] + tokens + [eos] for teacher forcing."""
        body = self.encode(text)
        ids = [self.bos_id] + body + [self.eos_id]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids


def train_sentencepiece(
    texts: list[str],
    model_prefix: Path,
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
) -> Path:
    """Train a joint BPE model; returns path to .model file."""
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = model_prefix.parent / f"{model_prefix.name}.corpus.txt"
    with corpus.open("w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(t.replace("\n", " ") + "\n")

    # user_defined_symbols keeps direction tags as atomic pieces.
    user_syms = ",".join(SPECIALS[4:])  # TAG_EN2ZH, TAG_ZH2EN
    # Chinese alphabets are large; bump vocab_size if SentencePiece rejects it.
    for attempt_vs, attempt_cov in (
        (vocab_size, character_coverage),
        (max(vocab_size, 8000), character_coverage),
        (max(vocab_size, 12000), 0.995),
    ):
        try:
            spm.SentencePieceTrainer.train(
                input=str(corpus),
                model_prefix=str(model_prefix),
                vocab_size=attempt_vs,
                model_type="bpe",
                character_coverage=attempt_cov,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece=PAD,
                unk_piece=UNK,
                bos_piece=BOS,
                eos_piece=EOS,
                user_defined_symbols=user_syms,
                num_threads=4,
                shuffle_input_sentence=True,
                input_sentence_size=min(2_000_000, max(10_000, len(texts))),
            )
            if attempt_vs != vocab_size:
                print(f"vocab_size 自动调整为 {attempt_vs}（中文字符集需要更大词表）")
            break
        except RuntimeError as e:
            if "Vocabulary size is smaller" not in str(e):
                raise
    else:
        raise RuntimeError("SentencePiece 训练失败：请增大 vocab_size")
    return Path(str(model_prefix) + ".model")
