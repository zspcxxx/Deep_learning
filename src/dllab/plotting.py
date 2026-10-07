from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
