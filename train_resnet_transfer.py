#!/usr/bin/env python3
"""Transfer learning for paired brightfield/fluorescence cell classification.

This script trains a two-stream model that uses two ImageNet-pretrained ResNet
backbones:
1. one backbone for the brightfield (BF) image
2. one backbone for the fluorescence (FL) image

The extracted features are concatenated and passed to a binary classifier to
predict the probability that a cell is cancerous.

Expected workspace layout:
.
├── BF/
│   ├── train/
│   └── test/
├── FL/
│   ├── train/
│   └── test/
├── train.csv
└── sampleSubmission.csv

Example:
    python3 train_resnet_transfer.py --epochs 12 --batch-size 32 --backbone resnet50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train.csv"
SAMPLE_SUBMISSION_CSV = ROOT / "sampleSubmission.csv"
BF_TRAIN_DIR = ROOT / "BF" / "train"
BF_TEST_DIR = ROOT / "BF" / "test"
FL_TRAIN_DIR = ROOT / "FL" / "train"
FL_TEST_DIR = ROOT / "FL" / "test"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Record:
    name: str
    label: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a paired BF/FL transfer learning model with ImageNet ResNet."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--checkpoint-name", type=str, default="best_resnet_transfer.pt")
    parser.add_argument("--submission-name", type=str, default="submission_resnet_transfer.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_train_records(csv_path: Path) -> List[Record]:
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        return [Record(name=row["Name"], label=float(row["Diagnosis"])) for row in reader]


def read_test_records(csv_path: Path) -> List[Record]:
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        return [Record(name=row["Name"], label=None) for row in reader]


def stratified_split(records: Sequence[Record], val_ratio: float, seed: int) -> Tuple[List[Record], List[Record]]:
    grouped: Dict[int, List[Record]] = {0: [], 1: []}
    for record in records:
        grouped[int(record.label)].append(record)

    rng = random.Random(seed)
    train_records: List[Record] = []
    val_records: List[Record] = []
    for label_records in grouped.values():
        label_records = label_records.copy()
        rng.shuffle(label_records)
        val_size = max(1, int(len(label_records) * val_ratio))
        val_records.extend(label_records[:val_size])
        train_records.extend(label_records[val_size:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


class PairedCellDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Record],
        bf_dir: Path,
        fl_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.records = list(records)
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        bf_image = self._load_image(self.bf_dir / record.name)
        fl_image = self._load_image(self.fl_dir / record.name)
        bf_tensor = self.transform(bf_image)
        fl_tensor = self.transform(fl_image)

        if record.label is None:
            return bf_tensor, fl_tensor, record.name

        label = torch.tensor(record.label, dtype=torch.float32)
        return bf_tensor, fl_tensor, label

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    resize_size = int(image_size * 1.14)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def get_backbone_and_weights(backbone_name: str):
    backbone_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
    }
    return backbone_map[backbone_name]


class TwoStreamResNet(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", dropout: float = 0.3) -> None:
        super().__init__()
        backbone_fn, weights, feature_dim = get_backbone_and_weights(backbone_name)
        self.bf_backbone = backbone_fn(weights=weights)
        self.fl_backbone = backbone_fn(weights=weights)
        self.bf_backbone.fc = nn.Identity()
        self.fl_backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, bf_images: torch.Tensor, fl_images: torch.Tensor) -> torch.Tensor:
        bf_features = self.bf_backbone(bf_images)
        fl_features = self.fl_backbone(fl_images)
        fused = torch.cat([bf_features, fl_features], dim=1)
        logits = self.classifier(fused).squeeze(1)
        return logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        for module in (self.bf_backbone, self.fl_backbone):
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def create_dataloaders(args: argparse.Namespace):
    train_records = read_train_records(TRAIN_CSV)
    train_split, val_split = stratified_split(train_records, args.val_ratio, args.seed)
    test_records = read_test_records(SAMPLE_SUBMISSION_CSV)

    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = PairedCellDataset(train_split, BF_TRAIN_DIR, FL_TRAIN_DIR, train_transform)
    val_dataset = PairedCellDataset(val_split, BF_TRAIN_DIR, FL_TRAIN_DIR, eval_transform)
    test_dataset = PairedCellDataset(test_records, BF_TEST_DIR, FL_TEST_DIR, eval_transform)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_split, val_split


def binary_auc_score(targets: Sequence[float], probabilities: Sequence[float]) -> float:
    targets_arr = np.asarray(targets, dtype=np.int64)
    probs_arr = np.asarray(probabilities, dtype=np.float64)

    positives = int(targets_arr.sum())
    negatives = int(len(targets_arr) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(probs_arr, kind="mergesort")
    sorted_probs = probs_arr[order]
    sorted_targets = targets_arr[order]

    rank_sum = 0.0
    start = 0
    n = len(sorted_probs)
    while start < n:
        end = start + 1
        while end < n and sorted_probs[end] == sorted_probs[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        positives_in_group = int(sorted_targets[start:end].sum())
        rank_sum += positives_in_group * avg_rank
        start = end

    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: List[float] = []
    all_targets: List[float] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    autocast_enabled = device.type == "cuda"

    for bf_images, fl_images, labels in loader:
        bf_images = bf_images.to(device, non_blocking=True)
        fl_images = fl_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(bf_images, fl_images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(np.int64)

        losses.append(float(loss.item()))
        all_probs.extend(probabilities.tolist())
        all_targets.extend(targets.tolist())
        all_preds.extend(predictions.tolist())

    accuracy = float(np.mean(np.asarray(all_preds) == np.asarray(all_targets, dtype=np.int64)))
    auc = binary_auc_score(all_targets, all_probs)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "auc": auc,
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], List[float]]:
    model.eval()
    names: List[str] = []
    probabilities: List[float] = []
    autocast_enabled = device.type == "cuda"

    with torch.no_grad():
        for bf_images, fl_images, batch_names in loader:
            bf_images = bf_images.to(device, non_blocking=True)
            fl_images = fl_images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(bf_images, fl_images)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            names.extend(batch_names)
            probabilities.extend(float(prob) for prob in probs)

    return names, probabilities


def save_submission(names: Sequence[str], probabilities: Sequence[float], output_path: Path) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Diagnosis"])
        writer.writerows(zip(names, probabilities))


def save_history(history: List[Dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "train_auc", "val_loss", "val_accuracy", "val_auc"],
        )
        writer.writeheader()
        writer.writerows(history)


def count_labels(records: Iterable[Record]) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for record in records:
        counts[int(record.label)] += 1
    return counts


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [
        TRAIN_CSV,
        SAMPLE_SUBMISSION_CSV,
        BF_TRAIN_DIR,
        BF_TEST_DIR,
        FL_TRAIN_DIR,
        FL_TEST_DIR,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset paths:\n" + "\n".join(missing))

    device = torch.device(args.device)
    train_loader, val_loader, test_loader, train_split, val_split = create_dataloaders(args)

    train_counts = count_labels(train_split)
    pos_weight_value = train_counts[0] / max(train_counts[1], 1)

    model = TwoStreamResNet(backbone_name=args.backbone, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_state = None
    best_val_auc = -math.inf
    history: List[Dict[str, float]] = []

    print(json.dumps(
        {
            "device": str(device),
            "backbone": args.backbone,
            "train_samples": len(train_split),
            "val_samples": len(val_split),
            "train_label_counts": train_counts,
            "val_label_counts": count_labels(val_split),
            "pos_weight": round(pos_weight_value, 4),
        },
        indent=2,
    ))

    for epoch in range(1, args.epochs + 1):
        backbone_trainable = epoch > args.freeze_backbone_epochs
        model.set_backbone_trainable(backbone_trainable)

        start_time = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device, scaler=None)
        scheduler.step()

        epoch_seconds = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f} | "
            f"backbone_trainable={backbone_trainable} | time={epoch_seconds:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
            }
        )

        val_auc = val_metrics["auc"]
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "best_val_auc": best_val_auc,
                "history": history,
            }
            checkpoint_path = args.output_dir / args.checkpoint_name
            torch.save(best_state, checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")

    if best_state is None:
        raise RuntimeError("Training finished without a valid validation AUC.")

    model.load_state_dict(best_state["model_state_dict"])
    test_names, test_probabilities = predict(model, test_loader, device)

    submission_path = args.output_dir / args.submission_name
    history_path = args.output_dir / "training_history.csv"
    save_submission(test_names, test_probabilities, submission_path)
    save_history(history, history_path)

    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Submission saved to: {submission_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
