from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .plotting import plt, save_figure


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders(data_dir: Path, batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def train(model, train_loader, test_loader, result_dir: Path, epochs=10, lr=1e-3, device=None, model_name="model"):
    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    print(f"\n{'=' * 70}\n开始训练 {model_name}\n{'=' * 70}")
    print(f"{'Epoch':>6s} | {'Train Loss':>11s} | {'Train Acc':>10s} | {'Test Loss':>10s} | {'Test Acc':>9s}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"{epoch:>6d} | {train_loss:>11.4f} | {train_acc:>9.2f}% | {test_loss:>10.4f} | {test_acc:>8.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), result_dir / f"{model_name}_best.pth")

    print("-" * 70)
    print(f"最佳测试准确率: {best_acc:.2f}%")
    return history, best_acc


def plot_training_history(history, model_name: str, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], "o-", color="tab:blue", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, history["test_loss"], "s-", color="tab:red", linewidth=2, label="Test Loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{model_name} - Loss Curve")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], "o-", color="tab:blue", linewidth=2, label="Train Acc")
    axes[1].plot(epochs, history["test_acc"], "s-", color="tab:red", linewidth=2, label="Test Acc")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"{model_name} - Accuracy Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    save_figure(fig, save_path)


@torch.no_grad()
def plot_confusion_matrix(model, loader, device, save_path: Path):
    model.eval()
    cm = np.zeros((10, 10), dtype=int)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, preds = model(images).max(1)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 50 else "black", fontsize=9)
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted Label", ylabel="True Label", title="Confusion Matrix (counts)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, save_path)


@torch.no_grad()
def plot_misclassified(model, loader, device, save_path: Path, n_show: int = 20):
    model.eval()
    wrong_images, wrong_labels, wrong_preds = [], [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        _, preds = outputs.max(1)
        mask = preds.cpu() != labels
        if mask.sum():
            wrong_images.append(images[mask])
            wrong_labels.append(labels[mask])
            wrong_preds.append(preds.cpu()[mask])
        if sum(len(w) for w in wrong_images) >= n_show:
            break
    if not wrong_images:
        return
    wrong_images = torch.cat(wrong_images)[:n_show]
    wrong_labels = torch.cat(wrong_labels)[:n_show].numpy()
    wrong_preds = torch.cat(wrong_preds)[:n_show].numpy()
    mean, std = 0.1307, 0.3081
    cols, rows = 5, (n_show + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_show):
        img = wrong_images[i].squeeze().numpy() * std + mean
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"True: {wrong_labels[i]}\nPred: {wrong_preds[i]}", fontsize=10, color="red")
    for i in range(n_show, len(axes)):
        axes[i].axis("off")
    fig.suptitle("Misclassified Samples", fontsize=14)
    save_figure(fig, save_path)


def plot_sample_images(loader, save_path: Path, n_show: int = 16):
    images, labels = next(iter(loader))
    cols, rows = 8, (n_show + 7) // 8
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_1d(axes).ravel()
    mean, std = 0.1307, 0.3081
    for i in range(n_show):
        img = images[i].squeeze().numpy() * std + mean
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Label: {labels[i].item()}", fontsize=10)
    for i in range(n_show, len(axes)):
        axes[i].axis("off")
    fig.suptitle("Sample Images from MNIST", fontsize=14)
    save_figure(fig, save_path)


@torch.no_grad()
def predict_single_image(model, image, device):
    model.eval()
    image = image.to(device)
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
    output = model(image)
    prob = torch.softmax(output, dim=1)
    pred = prob.argmax(dim=1).item()
    return pred, prob.squeeze().cpu().numpy()
