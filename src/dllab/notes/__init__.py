"""Annotations shown in the web runner: process notes, result interpretation, image captions."""
from __future__ import annotations

import traceback
from importlib import import_module
from pathlib import PurePosixPath

from dllab.notes.base import Annotator, LessonNotes, LogNote, finding

_MODULES = {
    "01": "l01_perceptron",
    "02": "l02_mlp",
    "03-visual": "l03_backprop_visual",
    "03": "l03_backpropagation",
    "04": "l04_activations",
    "05": "l05_learning_rate",
    "06": "l06_weight_init",
    "07": "l07_normalization",
    "08": "l08_regularization",
    "09": "l09_full_pipeline",
    "10": "l10_cnn",
    "11": "l11_residual",
    "12": "l12_rnn",
    "13": "l13_seq2seq",
    "14": "l14_attention",
}


def get_notes(lesson_id: str) -> LessonNotes:
    name = _MODULES.get(lesson_id)
    if name is None:
        return LessonNotes(intro="", steps=[], log_notes=[])
    return import_module(f"dllab.notes.{name}").NOTES


def explain_result(lesson_id: str, metrics: dict) -> list[dict]:
    try:
        return get_notes(lesson_id).explain(metrics)
    except Exception as exc:
        return [finding("结果解读生成失败", f"{exc}\n{traceback.format_exc(limit=2)}", "warn")]


def image_note(lesson_id: str, path: str) -> str:
    return get_notes(lesson_id).image_notes.get(PurePosixPath(path).name, "")


__all__ = ["Annotator", "LessonNotes", "LogNote", "explain_result", "finding", "get_notes", "image_note"]
