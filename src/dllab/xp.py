"""Array backend for training.

CuPy is used when CUDA is available, so the same handwritten NumPy code runs on the GPU.
Set DLLAB_GPU=0 to force CPU. Machines without CuPy stay on NumPy.
"""
from __future__ import annotations

import os

import numpy as _numpy

_flag = os.environ.get("DLLAB_GPU", "auto").strip().lower()
_cupy = None
if _flag not in {"0", "cpu", "off", "false"}:
    try:
        import cupy as _cupy_mod

        if _cupy_mod.cuda.runtime.getDeviceCount() > 0:
            _cupy = _cupy_mod
    except Exception:
        _cupy = None

GPU = _cupy is not None
# Matrices smaller than this are faster on CPU: a kernel launch costs more than the arithmetic.
_GPU_MIN_ELEMENTS = 262_144
_force_cpu = False

_usage = {"allocations": 0, "peak_bytes": 0}
if GPU:
    _pool = _cupy.get_default_memory_pool()

    def _counting_malloc(size):
        mem = _pool.malloc(size)
        _usage["allocations"] += 1
        _usage["peak_bytes"] = max(_usage["peak_bytes"], _pool.used_bytes())
        return mem

    _cupy.cuda.set_allocator(_counting_malloc)


def reset_gpu_usage() -> None:
    _usage.update(allocations=0, peak_bytes=0)


def gpu_usage() -> dict:
    """What actually ran on the GPU since reset_gpu_usage(): every CuPy array allocation is counted."""
    return {
        "available": GPU,
        "used": _usage["allocations"] > 0,
        "allocations": _usage["allocations"],
        "peak_mb": round(_usage["peak_bytes"] / 2**20, 1),
    }


def describe_usage(usage: dict) -> str:
    if not usage["available"]:
        return "GPU 使用: 否（本机没有可用的 CuPy / CUDA，全部用 NumPy 在 CPU 上计算）"
    if not usage["used"]:
        return f"GPU 使用: 否（没有超过 {_GPU_MIN_ELEMENTS} 个数的矩阵，全部在 CPU 上计算）"
    return f"GPU 使用: 是（CuPy 显存分配 {usage['allocations']} 次，峰值 {usage['peak_mb']} MB）"


def on_gpu() -> bool:
    return GPU and not _force_cpu


class _Np:
    """NumPy or CuPy, chosen when each operation runs."""

    def __getattr__(self, name):
        return getattr(_cupy if on_gpu() else _numpy, name)


np = _Np()
# Initial parameters are built on the host; `np` here would allocate on the GPU first.
host_np = _numpy


class prefer_cpu_if_small:
    """Use NumPy for a small training set. Large batches keep using the GPU."""

    def __init__(self, n_elements: int):
        self._prev = False
        self.n_elements = int(n_elements)

    def __enter__(self):
        global _force_cpu
        self._prev = _force_cpu
        if GPU and self.n_elements < _GPU_MIN_ELEMENTS:
            _force_cpu = True
        return on_gpu()

    def __exit__(self, *exc):
        global _force_cpu
        _force_cpu = self._prev
        return False


def force_cpu() -> prefer_cpu_if_small:
    """NumPy float64 whatever the size: finite-difference checks are meaningless in float32."""
    return prefer_cpu_if_small(0)


def asnumpy(a):
    if on_gpu():
        return _cupy.asnumpy(a)
    return _numpy.asarray(a)


def host_rng(seed: int):
    """NumPy RNG so initial weights match a CPU run. CuPy's generator is a different sequence."""
    return _numpy.random.RandomState(int(seed))


def as_param(values, dtype=None):
    """Parameter array on the host. CuPy values are copied back first."""
    getter = getattr(values, "get", None)
    if callable(getter) and type(values).__module__.startswith("cupy"):
        src = _numpy.asarray(getter())
    else:
        src = _numpy.asarray(values)
    if dtype is not None:
        src = src.astype(getattr(dtype, "name", dtype), copy=False)
    return src


def as_device(a):
    """Move to the compute device without changing dtype (index and label arrays)."""
    return np.asarray(a)


def as_compute(a):
    """Floating-point array on the compute device. GPU training uses float32."""
    if on_gpu():
        return np.asarray(a, dtype=np.float32)
    return np.asarray(a)


def cast_params(pairs) -> None:
    """Cast parameter arrays owned by (key, obj, name) triples to the compute dtype."""
    if not on_gpu():
        return
    for _, obj, name in pairs:
        setattr(obj, name, np.asarray(getattr(obj, name), dtype=np.float32))


def cast_lists(*groups) -> list:
    if not on_gpu():
        return list(groups)
    return [[np.asarray(v, dtype=np.float32) for v in group] for group in groups]


def move_params(pairs, *, host: bool) -> None:
    if not on_gpu():
        return
    for _, obj, name in list(pairs):
        value = getattr(obj, name)
        if host:
            setattr(obj, name, _numpy.array(asnumpy(value), copy=True))
        else:
            setattr(obj, name, np.asarray(value))


def move_lists(obj, names: tuple[str, ...], *, host: bool) -> None:
    if not on_gpu():
        return
    for name in names:
        if not hasattr(obj, name):
            continue
        values = getattr(obj, name)
        if not values:
            continue
        if host:
            setattr(obj, name, [_numpy.array(asnumpy(v), copy=True) for v in values])
        else:
            setattr(obj, name, [np.asarray(v) for v in values])


def gpu_enter(obj, mover) -> None:
    depth = getattr(obj, "_gpu_depth", 0)
    if depth == 0:
        mover(host=False)
    obj._gpu_depth = depth + 1


def gpu_exit(obj, mover) -> None:
    obj._gpu_depth = max(0, getattr(obj, "_gpu_depth", 1) - 1)
    if obj._gpu_depth == 0:
        mover(host=True)


def describe() -> str:
    if not GPU:
        return "计算设备: CPU（NumPy）"
    name = _cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return (
        f"计算设备: GPU（CuPy / {name}，大矩阵用 float32；"
        f"不足 {_GPU_MIN_ELEMENTS} 个数的小网络用 CPU）"
    )
