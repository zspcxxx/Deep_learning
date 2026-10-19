from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_CJK_FONTS = ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei")
_installed = {f.name for f in font_manager.fontManager.ttflist}
# Per-glyph fallback (matplotlib >= 3.6) only walks the families listed in font.family itself;
# a generic "sans-serif" alias resolves to a single font. Listing only installed fonts avoids "findfont" noise.
plt.rcParams["font.family"] = ["DejaVu Sans", *[f for f in _CJK_FONTS if f in _installed]]
plt.rcParams["axes.unicode_minus"] = False
# Hosts without any CJK font would otherwise print one warning per missing glyph per figure.
warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font")
# Gridspec figures with fig.text; save_figure crops with bbox_inches="tight" anyway.
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 110) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decision_boundary(ax, model, X, y, title: str) -> None:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2, linestyles="--")

    pos, neg = y == 1, y == 0
    ax.scatter(
        X[pos, 0], X[pos, 1], c="red", marker="o", s=18, alpha=0.7,
        edgecolor="k", linewidth=0.4, label="Class 1",
    )
    ax.scatter(
        X[neg, 0], X[neg, 1], c="blue", marker="x", s=18, alpha=0.7, label="Class 0",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.3)
