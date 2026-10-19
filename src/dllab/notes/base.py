from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LogNote:
    """Annotation attached to the first (or every) log line matching `pattern`.

    Named groups in the pattern can be referenced as {name} in title / body.
    `step` (0-based) marks that step of LessonNotes.steps as reached.
    """

    pattern: str
    title: str
    body: str
    step: int | None = None
    once: bool = True
    level: str = "info"  # info | ok | warn | error


@dataclass
class LessonNotes:
    intro: str
    steps: list[str]
    log_notes: list[LogNote]
    image_notes: dict[str, str] = field(default_factory=dict)
    explain: Callable[[dict], list[dict]] = lambda metrics: []


def finding(title: str, body: str, level: str = "info") -> dict:
    return {"title": title, "body": body, "level": level}


def pct(x: float) -> str:
    return f"{x:.2f}%"


COMMON_LOG_NOTES = [
    LogNote(
        r"Epoch +\d+ \| loss = [\d.]+ \| accuracy",
        "怎么读训练日志",
        "每行是一次全批量迭代（epoch）后的状态：loss 是训练集上的二元交叉熵（越小越好，"
        "0.693≈ln2 表示模型对每个样本都输出 0.5，等于没学到东西），accuracy 是训练集准确率。"
        "只打印每隔若干轮的一行，中间的迭代照常进行。",
    ),
    LogNote(
        r"Epoch +\d+/\d+ \| lr = ",
        "怎么读 mini-batch 训练日志",
        "每行是一个 epoch（把训练集按 batch 完整过一遍）结束后的状态：lr 是当前学习率（受 Warmup + Cosine 调度控制），"
        "loss / train_acc 是本轮各 batch 的平均训练损失与准确率（训练模式，含 Dropout），"
        "val_loss / val_acc 是在验证集上（推理模式）的结果——早停和模型选择看的是验证集。",
    ),
    LogNote(
        r"Early stop at epoch (?P<stop>\d+) \(best=(?P<best>\d+)",
        "触发早停（第 {stop} 轮）",
        "验证损失已经连续若干轮（patience）没有创新低，继续训练只会让模型更贴合训练集。"
        "于是在第 {stop} 轮停止，并把参数回滚到验证损失最低的第 {best} 轮——最终评估用的是那一轮的模型。",
        once=False,
        level="ok",
    ),
    LogNote(
        r"结果(图)?已保存",
        "结果已写入 artifacts/",
        "图片和 metrics.json 已写到 artifacts/ 下对应目录，运行结束后会显示在下方“运行结果”区。",
    ),
    LogNote(
        r"^Traceback \(most recent call last\)",
        "运行出错",
        "下面是 Python 异常栈：从上往下是调用链，最后一行是异常类型和原因，通常先看最后一行。",
        level="error",
    ),
]


class Annotator:
    """Turn a stream of log lines into annotation events for one run."""

    def __init__(self, notes: LessonNotes):
        self._rules = [(re.compile(n.pattern), n) for n in [*notes.log_notes, *COMMON_LOG_NOTES]]
        self._seen: set[int] = set()

    def feed(self, line: str) -> list[dict]:
        out = []
        for i, (rx, note) in enumerate(self._rules):
            m = rx.search(line)
            if not m or (note.once and i in self._seen):
                continue
            self._seen.add(i)
            groups = m.groupdict()
            title = note.title.format(**groups) if groups else note.title
            body = note.body.format(**groups) if groups else note.body
            out.append({"title": title, "body": body, "level": note.level, "step": note.step})
        return out
