"""Plain vs residual deep MLPs with the same layers, for the residual-connection lesson.

Both share a stem Dense(n_in -> width) and a head ReLU -> Dense(width -> n_classes).
Each of the `n_blocks` body blocks computes  F(h) = W2 · relu(W1 · relu(h) + b1) + b2  (pre-activation):

    plain     h <- F(h)
    residual  h <- h + F(h)

so the two models have exactly the same parameters and differ only in the skip connection.
"""
from __future__ import annotations

from dllab.nn import Dense, ReLU, Residual, Sequential, softmax
from dllab.xp import gpu_enter, gpu_exit, host_rng, np


def build_deep_mlp(
    n_blocks: int,
    residual: bool,
    width: int = 32,
    n_in: int = 2,
    n_classes: int = 3,
    branch_init: str = "zeros",
    seed: int = 42,
    dtype=np.float64,
) -> Sequential:
    """`branch_init` is the init of the last Dense in each residual branch ("zeros" = block starts as identity)."""
    rng = host_rng(seed)
    layers, names = [Dense(n_in, width, rng, dtype)], ["stem"]
    for k in range(1, n_blocks + 1):
        branch = [
            ("relu_a", ReLU()),
            ("fc_a", Dense(width, width, rng, dtype)),
            ("relu_b", ReLU()),
            ("fc_b", Dense(width, width, rng, dtype, init=branch_init if residual else "he")),
        ]
        if residual:
            layers.append(Residual(branch))
            names.append(f"block{k}")
        else:
            for cname, layer in branch:
                layers.append(layer)
                names.append(f"block{k}.{cname}")
    layers += [ReLU(), Dense(width, n_classes, rng, dtype, init="xavier")]
    names += ["head_relu", "head"]
    return Sequential(layers, names, dtype)


def n_weight_layers(n_blocks: int) -> int:
    """Dense layers on the longest path: stem + 2 per block + head."""
    return 2 * n_blocks + 2


def block_outputs(model: Sequential, x) -> list[np.ndarray]:
    """Hidden state h after the stem and after every block (inference mode)."""
    gpu_enter(model, model._move)
    try:
        a = np.asarray(x, dtype=model.dtype)
        outs = []
        for name, layer in zip(model.layer_names, model.layers):
            a = layer.forward(a, False)
            if name == "stem" or isinstance(layer, Residual) or name.endswith(".fc_b"):
                outs.append(a)
        return outs
    finally:
        gpu_exit(model, model._move)


def block_input_grads(model: Sequential, x, y) -> list[float]:
    """‖dL/dh‖ at the input of every block (block 1 first), from one backward pass at the current weights."""
    gpu_enter(model, model._move)
    try:
        y = np.asarray(y)
        probs = softmax(model.forward(x, training=True))
        d = probs.copy()
        d[np.arange(len(y)), y] -= 1.0
        d /= len(y)
        norms = []
        for name, layer in reversed(list(zip(model.layer_names, model.layers))):
            d = layer.backward(d)
            if isinstance(layer, Residual) or name.endswith(".relu_a"):
                norms.append(float(np.linalg.norm(d)))
        return norms[::-1]
    finally:
        gpu_exit(model, model._move)
