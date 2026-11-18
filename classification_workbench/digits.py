"""Train and compare fully connected and convolutional digit classifiers."""

from dataclasses import asdict, dataclass, replace
from math import ceil, floor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from .reporting import draw_confusion, multiclass_scores, save_figure, write_metrics


MEAN, STD = 0.1307, 0.3081
LEARNING_RATE = 0.001
ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


@dataclass(frozen=True)
class Settings:
    batch_size: int = 64
    hidden_widths: tuple[int, ...] = (128,)
    activation: str = "relu"
    dropout: float = 0.0
    augment: bool = False
    architecture: str = "mlp"


def candidates(sweep, extensions):
    baseline = Settings()
    groups = {"baseline": {"baseline": baseline}}
    if sweep:
        groups.update({
            "batch size": {
                f"batch-{value}": replace(baseline, batch_size=value) for value in (32, 128)
            },
            "depth": {
                f"depth-{value}": replace(baseline, hidden_widths=(128,) * value) for value in (3, 5)
            },
            "width": {
                f"width-{value}": replace(baseline, hidden_widths=(value,)) for value in (64, 256, 512)
            },
            "activation": {
                name: replace(baseline, activation=name) for name in ("sigmoid", "tanh", "leaky_relu")
            },
            "dropout": {
                f"dropout-{value}": replace(baseline, dropout=value) for value in (0.2, 0.5)
            },
        })
    if extensions:
        groups.update({
            "joint tuning": {
                "batch-32-width-256": replace(baseline, batch_size=32, hidden_widths=(256,)),
                "width-256-dropout-0.2": replace(baseline, hidden_widths=(256,), dropout=0.2),
            },
            "augmentation": {"augmented-mlp": replace(baseline, augment=True)},
            "architecture": {"small-cnn": replace(baseline, architecture="cnn", hidden_widths=())},
        })
    return groups


def load_data(args):
    generator = torch.Generator().manual_seed(args.seed)
    if args.demo:
        digits = load_digits()
        images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1)
        images = F.interpolate(images, size=(28, 28), mode="bilinear", align_corners=False)
        dataset = TensorDataset(images * (255 / 16), torch.tensor(digits.target, dtype=torch.long))
        return random_split(dataset, [0.7, 0.15, 0.15], generator=generator)

    from torchvision.datasets import MNIST

    root = args.data_dir or Path("data/digits")
    original_train = MNIST(root, train=True, download=True)
    original_test = MNIST(root, train=False, download=True)
    train_full = TensorDataset(original_train.data.unsqueeze(1), original_train.targets)
    train, validation = random_split(train_full, [50_000, 10_000], generator=generator)
    test = TensorDataset(original_test.data.unsqueeze(1), original_test.targets)
    return train, validation, test


def build_model(settings):
    if settings.architecture == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    layers = [nn.Flatten()]
    input_width = 28 * 28
    for width in settings.hidden_widths:
        layers.extend((nn.Linear(input_width, width), ACTIVATIONS[settings.activation]()))
        if settings.dropout > 0:
            layers.append(nn.Dropout(settings.dropout))
        input_width = width
    layers.append(nn.Linear(input_width, 10))
    return nn.Sequential(*layers)


def augment_batch(images):
    """Apply an independent small rotation and translation to each training image."""
    batch_size = images.size(0)
    angles = torch.empty(batch_size, device=images.device).uniform_(-10, 10).deg2rad()
    shifts = torch.empty(batch_size, 2, device=images.device).uniform_(-2, 2) * (2 / 28)
    cos, sin = angles.cos(), angles.sin()
    affine = torch.stack((cos, -sin, shifts[:, 0], sin, cos, shifts[:, 1]), dim=1)
    grid = F.affine_grid(affine.reshape(-1, 2, 3), images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def prepare_images(images, device, augment=False):
    images = images.to(device=device, dtype=torch.float32) / 255
    if augment:
        images = augment_batch(images)
    return (images - MEAN) / STD


def train_epoch(model, loader, optimizer, loss_function, device, augment):
    model.train()
    loss_sum, correct = 0.0, 0
    for images, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(prepare_images(images, device, augment=augment))
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, loss_function, device):
    model.eval()
    loss_sum = 0.0
    actual, predicted = [], []
    for images, labels in loader:
        logits = model(prepare_images(images, device))
        loss_sum += loss_function(logits, labels.to(device)).item() * labels.numel()
        actual.append(labels.numpy())
        predicted.append(logits.argmax(dim=1).cpu().numpy())
    return loss_sum / len(loader.dataset), np.concatenate(actual), np.concatenate(predicted)


def train(settings, train_set, validation_set, args, device, loss_function):
    torch.manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, generator=shuffle)
    val_loader = DataLoader(validation_set, batch_size=settings.batch_size)
    model = build_model(settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, loss_function, device, settings.augment
        )
        val_loss, actual, predicted = evaluate(model, val_loader, loss_function, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": float(np.mean(actual == predicted)),
            "seconds": perf_counter() - started,
        })
    return model, history


def plot_learning_curves(runs, winner, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    names = ["baseline"] if winner == "baseline" else ["baseline", winner]
    for name in names:
        history = runs[name]["history"]
        epochs = [row["epoch"] for row in history]
        for axis, metric in zip(axes, ("loss", "accuracy")):
            line, = axis.plot(epochs, [row[f"validation_{metric}"] for row in history],
                              marker="o", label=f"{name} validation")
            axis.plot(epochs, [row[f"train_{metric}"] for row in history],
                      linestyle="--", color=line.get_color(), alpha=0.75, label=f"{name} train")
            axis.set(xlabel="Epoch", ylabel=metric.capitalize())
            axis.set_xticks(epochs if len(epochs) <= 10 else axis.get_xticks())
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    save_figure(fig, out_dir / "learning-curves.png")


def plot_comparison(runs, winner, out_dir):
    names = sorted(runs, key=lambda name: runs[name]["history"][-1]["validation_accuracy"])
    accuracy = [100 * runs[name]["history"][-1]["validation_accuracy"] for name in names]
    colors = ["#e2824c" if name == winner else "#326684" for name in names]
    fig, axis = plt.subplots(figsize=(8, 1.5 + 0.34 * len(names)), layout="constrained")
    axis.scatter(accuracy, names, color=colors, s=34, zorder=3)
    for name, score in zip(names, accuracy):
        axis.annotate(f"{score:.2f}", (score, name), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=8)
    padding = max(1, 0.12 * (max(accuracy) - min(accuracy)))
    axis.set(xlabel="Final-epoch validation accuracy (%)",
             xlim=(max(0, floor(min(accuracy) - padding)), ceil(max(accuracy) + padding)))
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right", "left"]].set_visible(False)
    save_figure(fig, out_dir / "validation-comparison.png")


def run(args):
    device_name = args.device
    cuda_available = torch.cuda.is_available()
    if device_name == "auto":
        device_name = "cuda" if cuda_available else "cpu"
    if device_name == "cuda" and not cuda_available:
        raise ValueError("CUDA is unavailable. Use --device cpu or a CUDA-enabled PyTorch install.")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    train_set, validation_set, test_set = load_data(args)
    loss_function = nn.CrossEntropyLoss()
    runs = {}
    winner, best_accuracy = "baseline", -1.0
    for group, configurations in candidates(args.sweep, args.extensions).items():
        for name, settings in configurations.items():
            model, history = train(settings, train_set, validation_set, args, device, loss_function)
            accuracy = history[-1]["validation_accuracy"]
            runs[name] = {
                "group": group,
                "settings": asdict(settings),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "history": history,
            }
            print(f"{name}: validation accuracy={accuracy:.2%}", flush=True)
            if accuracy > best_accuracy:
                winner, best_accuracy = name, accuracy
                selected_model, selected_settings = model, settings

    test_loader = DataLoader(test_set, batch_size=selected_settings.batch_size)
    test_loss, actual, predicted = evaluate(selected_model, test_loader, loss_function, device)
    labels = list(range(10))
    test_scores = multiclass_scores(actual, predicted, labels)
    confusion = confusion_matrix(actual, predicted, labels=labels)
    report = {
        "dataset": "scikit-learn digits (resized)" if args.demo else "MNIST",
        "seed": args.seed,
        "device": device.type,
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
        "epochs": args.epochs,
        "split_sizes": {"train": len(train_set), "validation": len(validation_set), "test": len(test_set)},
        "runs": runs,
        "selected_run": winner,
        "test": {"loss": test_loss, **test_scores, "confusion_matrix": confusion.tolist()},
    }
    write_metrics(args.out, report)
    print(f"{winner}: test accuracy={test_scores['accuracy']:.2%}, macro-F1={test_scores['macro_f1']:.4f}")

    plot_learning_curves(runs, winner, args.out)
    if len(runs) > 1:
        plot_comparison(runs, winner, args.out)
    fig, axis = plt.subplots(figsize=(6, 5.5), layout="constrained")
    draw_confusion(axis, confusion, labels, f"{winner} on the test split")
    save_figure(fig, args.out / "confusion.png")
