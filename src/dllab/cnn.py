"""From-scratch convolutional network (NumPy only, NCHW layout).

Conv2D uses im2col so that convolution becomes one matrix multiply; its backward pass
reuses the same columns for dW and scatters dcols back with col2im.
Generic layers, the training loop and gradient_check live in dllab.nn.
"""
from __future__ import annotations

from dllab.nn import Dense, Dropout, Flatten, ReLU, Sequential, gradient_check
from dllab.xp import as_param, asnumpy, gpu_enter, gpu_exit, host_np, host_rng, np

__all__ = ["Conv2D", "MaxPool2", "SimpleCNN", "augment_flip_crop", "col2im", "gradient_check", "im2col"]


def im2col(x: np.ndarray, k: int, pad: int):
    """(N, C, H, W) -> (N*Ho*Wo, C*k*k) patch matrix for a stride-1 k x k window."""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    ho, wo = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    s0, s1, s2, s3 = xp.strides
    kwargs = {} if type(xp).__module__.startswith("cupy") else {"writeable": False}
    windows = np.lib.stride_tricks.as_strided(
        xp, shape=(n, c, k, k, ho, wo), strides=(s0, s1, s2, s3, s2, s3), **kwargs
    )
    cols = windows.transpose(0, 4, 5, 1, 2, 3).reshape(n * ho * wo, c * k * k)
    return cols, ho, wo


def col2im(dcols: np.ndarray, x_shape: tuple, k: int, pad: int, ho: int, wo: int) -> np.ndarray:
    """Inverse scatter of im2col: overlapping patches accumulate their gradients."""
    n, c, h, w = x_shape
    d = dcols.reshape(n, ho, wo, c, k, k).transpose(0, 3, 4, 5, 1, 2)
    dxp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=dcols.dtype)
    for i in range(k):
        for j in range(k):
            dxp[:, :, i : i + ho, j : j + wo] += d[:, :, i, j]
    return dxp[:, :, pad : pad + h, pad : pad + w] if pad else dxp


class Conv2D:
    param_names = ("W", "b")

    def __init__(self, c_in: int, c_out: int, k: int, pad: int, rng, dtype):
        self.k, self.pad = k, pad
        self.W = as_param(rng.randn(c_out, c_in, k, k) * (2.0 / (c_in * k * k)) ** 0.5, dtype)
        self.b = as_param(host_np.zeros(c_out), dtype)

    def forward(self, x, training: bool):
        cols, ho, wo = im2col(x, self.k, self.pad)
        self._cache = (x.shape, cols, ho, wo)
        f = self.W.shape[0]
        out = cols @ self.W.reshape(f, -1).T + self.b
        return out.reshape(x.shape[0], ho, wo, f).transpose(0, 3, 1, 2)

    def backward(self, dout):
        x_shape, cols, ho, wo = self._cache
        f = self.W.shape[0]
        d = dout.transpose(0, 2, 3, 1).reshape(-1, f)
        self.dW = (d.T @ cols).reshape(self.W.shape)
        self.db = d.sum(axis=0)
        return col2im(d @ self.W.reshape(f, -1), x_shape, self.k, self.pad, ho, wo)


class MaxPool2:
    """2x2 max pooling, stride 2. The gradient goes only to the arg-max of each window."""

    param_names = ()

    def forward(self, x, training: bool):
        n, c, h, w = x.shape
        ho, wo = h // 2, w // 2
        win = x.reshape(n, c, ho, 2, wo, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, ho, wo, 4)
        idx = win.argmax(axis=-1)
        self._cache = (x.shape, idx)
        return np.take_along_axis(win, idx[..., None], axis=-1)[..., 0]

    def backward(self, dout):
        (n, c, h, w), idx = self._cache
        ho, wo = h // 2, w // 2
        dwin = np.zeros((n, c, ho, wo, 4), dtype=dout.dtype)
        np.put_along_axis(dwin, idx[..., None], dout[..., None], axis=-1)
        return dwin.reshape(n, c, ho, wo, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


def augment_flip_crop(x, rng, pad: int = 2):
    """Random horizontal flip + random crop after zero padding (NCHW batch)."""
    n, _, h, w = x.shape
    flip = np.asarray(rng.rand(n) < 0.5)
    x = np.where(flip[:, None, None, None], x[..., ::-1], x)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dy, dx = asnumpy(rng.randint(0, 2 * pad + 1, size=(2, n)))
    return np.stack([xp[i, :, int(dy[i]) : int(dy[i]) + h, int(dx[i]) : int(dx[i]) + w] for i in range(n)])


class SimpleCNN(Sequential):
    """[conv3x3 -> ReLU -> maxpool2] x len(channels) -> FC(hidden) -> ReLU -> Dropout -> FC(n_classes)."""

    def __init__(
        self,
        channels: tuple[int, ...] = (8, 16),
        hidden: int = 64,
        n_classes: int = 10,
        image_size: int = 28,
        in_channels: int = 1,
        dropout: float = 0.2,
        seed: int = 42,
        dtype=np.float32,
    ):
        if image_size % 2 ** len(channels):
            raise ValueError(f"image_size must be divisible by {2 ** len(channels)} ({len(channels)} 2x2 poolings)")
        rng = host_rng(seed)
        layers, names = [], []
        c_prev = in_channels
        for k, c in enumerate(channels, start=1):
            layers += [Conv2D(c_prev, c, 3, 1, rng, dtype), ReLU(), MaxPool2()]
            names += [f"conv{k}", f"relu{k}", f"pool{k}"]
            c_prev = c
        flat = c_prev * (image_size // 2 ** len(channels)) ** 2
        layers += [
            Flatten(),
            Dense(flat, hidden, rng, dtype),
            ReLU(),
            Dropout(dropout, seed + 7),
            Dense(hidden, n_classes, rng, dtype, init="xavier"),
        ]
        names += ["flatten", "fc1", "relu_fc", "dropout", "fc2"]
        super().__init__(layers, names, dtype)
        self.n_conv = len(channels)

    def feature_maps(self, x) -> dict[str, np.ndarray]:
        """Activations after each conv block's ReLU (relu1, relu2, ...) for visualization."""
        gpu_enter(self, self._move)
        try:
            a = np.asarray(x, dtype=self.dtype)
            maps = {}
            last = f"relu{self.n_conv}"
            for name, layer in zip(self.layer_names, self.layers):
                a = layer.forward(a, False)
                if name.startswith("relu") and name[4:].isdigit():
                    maps[name] = asnumpy(a)
                if name == last:
                    break
            return maps
        finally:
            gpu_exit(self, self._move)
