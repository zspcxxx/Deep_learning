"""Full bilingual English ↔ Chinese Transformer machine translation."""
import os

# PyTorch 2.14 hands some eager matmuls (e.g. the 1×1 outer product in the first decoding step)
# to Triton kernels, which compile a helper against Python.h at runtime and fail on systems
# without python3-dev. cuBLAS is just as fast at this model size. Must be set before torch loads.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

__version__ = "0.1.0"
