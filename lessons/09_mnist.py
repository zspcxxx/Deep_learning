#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from dllab.io_utils import dump_metrics
from dllab.mnist import (
    CNN, MLP, get_data_loaders, get_device, plot_confusion_matrix,
    plot_misclassified, plot_sample_images, plot_training_history,
    predict_single_image, train,
)
from dllab.paths import ensure_artifacts, PROJECT_ROOT


def run(epochs: int = 3, batch_size: int = 256, lr: float = 1e-3) -> dict:
    out = ensure_artifacts("09_mnist")
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)
    print(f"训练集: {len(train_loader.dataset)}  测试集: {len(test_loader.dataset)}")
    plot_sample_images(train_loader, out / "samples.png")

    mlp = MLP()
    print(f"MLP 参数量: {sum(p.numel() for p in mlp.parameters()):,}")
    history_mlp, acc_mlp = train(mlp, train_loader, test_loader, out, epochs=epochs, lr=lr, device=device, model_name="mlp")
    plot_training_history(history_mlp, "MLP", out / "mlp_history.png")

    cnn = CNN()
    print(f"CNN 参数量: {sum(p.numel() for p in cnn.parameters()):,}")
    history_cnn, acc_cnn = train(cnn, train_loader, test_loader, out, epochs=epochs, lr=lr, device=device, model_name="cnn")
    plot_training_history(history_cnn, "CNN", out / "cnn_history.png")

    best_cnn = CNN().to(device)
    best_cnn.load_state_dict(torch.load(out / "cnn_best.pth", map_location=device, weights_only=True))
    plot_confusion_matrix(best_cnn, test_loader, device, out / "confusion_matrix.png")
    plot_misclassified(best_cnn, test_loader, device, out / "misclassified.png")

    samples = []
    images, labels = next(iter(test_loader))
    for i in range(5):
        pred, prob = predict_single_image(best_cnn, images[i], device)
        true_label = labels[i].item()
        samples.append({"true": int(true_label), "pred": int(pred), "confidence": float(prob[pred])})
        print(f"  样本 {i+1}: 真实={true_label}, 预测={pred}, 置信度={prob[pred]:.4f}")

    images = [
        "artifacts/09_mnist/samples.png",
        "artifacts/09_mnist/mlp_history.png",
        "artifacts/09_mnist/cnn_history.png",
        "artifacts/09_mnist/confusion_matrix.png",
        "artifacts/09_mnist/misclassified.png",
    ]
    metrics = {
        "device": str(device),
        "mlp_best_acc": acc_mlp,
        "cnn_best_acc": acc_cnn,
        "history_mlp": history_mlp,
        "history_cnn": history_cnn,
        "samples": samples,
        "images": images,
        "passed": acc_mlp >= 95 and acc_cnn >= 97,
    }
    dump_metrics(out / "metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run(epochs=args.epochs, batch_size=args.batch_size)
