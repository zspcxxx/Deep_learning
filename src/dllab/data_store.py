"""Per-lesson dataset cache: data/<lesson>/<name>.npz.

Each file stores the arrays returned by a dataset builder plus the call signature
(builder name + arguments). A file is reused only when the signature matches, so
changing a lesson's data parameters regenerates it instead of silently loading stale data.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from dllab.paths import ensure_data


def _signature(builder: Callable, args: tuple, kwargs: dict) -> str:
    def default(obj: Any):
        if isinstance(obj, (np.ndarray, np.generic)):
            return np.asarray(obj).tolist()
        return str(obj)

    payload = {"builder": builder.__name__, "args": list(args), "kwargs": kwargs}
    return json.dumps(payload, sort_keys=True, default=default)


def cached_dataset(lesson: str, name: str, builder: Callable[..., tuple], *args, **kwargs) -> tuple:
    path = ensure_data(lesson) / f"{name}.npz"
    sig = _signature(builder, args, kwargs)
    if path.exists():
        with np.load(path, allow_pickle=False) as f:
            if "__signature__" in f and str(f["__signature__"]) == sig:
                return tuple(f[f"arr_{i}"] for i in range(int(f["__count__"])))

    arrays = tuple(np.asarray(a) for a in builder(*args, **kwargs))
    np.savez_compressed(
        path,
        __signature__=np.array(sig),
        __count__=np.array(len(arrays)),
        **{f"arr_{i}": a for i, a in enumerate(arrays)},
    )
    return arrays
