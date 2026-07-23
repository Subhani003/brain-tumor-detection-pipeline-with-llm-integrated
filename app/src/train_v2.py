"""
v2 training script — trains 3 architectures sequentially on the combined
brisc2025 + Epic/CSCR + Kaggle dataset (14 095 unique images after MD5 dedup).

Models trained (in order):
    1. ConvNeXt-Tiny       (~28 M params, replaces the old DenseNet-169)
    2. EfficientNet-B3     (~12 M params, was the strongest in v1)
    3. ResNet-50           (~25 M params, kept for ensemble diversity)

Training recipe (same across all three):
    Two-phase fine-tuning from ImageNet weights:
      Phase 1: backbone frozen, only the new classifier head learns (5 epochs).
      Phase 2: unfreeze everything, cosine-anneal LR, AMP + label smoothing
               + class-balanced weighted sampling, mixup with prob 0.2.
    Augmentations (Albumentations):
      - CLAHE p=0.5  (clipLimit=2.0, tileGrid=8x8 — same params as the
                      inference-time CLAHE in app.py, so the model sees the
                      same distribution at both ends)
      - Horizontal flip p=0.5  (tumor type does not depend on L/R)
      - Rotation ±15°
      - Brightness/contrast ±20%
      - ElasticTransform mild  (simulates inter-scanner deformation)
      - Gaussian noise sigma=0.01
    Validation uses NO augmentation — only Resize + ImageNet normalize.

Outputs:
    models/v2/convnext_tiny.pth      (best-val checkpoint, full state_dict)
    models/v2/efficientnet_b3.pth
    models/v2/resnet50.pth
    reports/train_v2_<model>_history.csv

Reproducibility: SEED = 42 (matches the split).

Run:
    python src/train_v2.py
    python src/train_v2.py --model convnext_tiny --epochs 40 --batch 32

Estimated GPU time on RTX 4060 Laptop, bs=32, AMP on:
    ConvNeXt-Tiny      ~50-60 min
    EfficientNet-B3    ~30-40 min
    ResNet-50          ~35-45 min
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import (
    ConvNeXt_Tiny_Weights, EfficientNet_B3_Weights, ResNet50_Weights,
    convnext_tiny, efficientnet_b3, resnet50,
)

# ── Paths ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "dataset"
OUT_DIR   = ROOT / "app" / "models" / "v2"
REPORTS   = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── Hyperparams ────────────────────────────────────────────────────
CLASSES   = ("glioma", "meningioma", "notumor", "pituitary")
NUM_CLS   = len(CLASSES)
SEED      = 42
IMG_SIZE  = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Per-model training config. Each model gets its own LR / weight decay tuned
# for its architecture family. Numbers picked from each paper's recommended
# fine-tuning recipe, scaled for our smaller dataset.
MODEL_CONFIGS = {
    "convnext_tiny": {
        "factory":    convnext_tiny,
        "weights":    ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "head_attr":  "classifier",  # nn.Sequential, last layer is Linear at idx -1
        "lr_head":    1e-3,
        "lr_full":    2e-5,
        "weight_decay": 0.05,
        "ema_decay":  0.9999,
    },
    "efficientnet_b3": {
        "factory":    efficientnet_b3,
        "weights":    EfficientNet_B3_Weights.IMAGENET1K_V1,
        "head_attr":  "classifier",
        "lr_head":    1e-3,
        "lr_full":    3e-5,
        "weight_decay": 0.01,
        "ema_decay":  0.999,
    },
    "resnet50": {
        "factory":    resnet50,
        "weights":    ResNet50_Weights.IMAGENET1K_V2,
        "head_attr":  "fc",
        "lr_head":    1e-3,
        "lr_full":    3e-5,
        "weight_decay": 0.01,
        "ema_decay":  0.999,
    },
}


# ── Dataset ────────────────────────────────────────────────────────
class TumorDataset(Dataset):
    def __init__(self, root: Path, transform: A.Compose):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file():
                    self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = np.array(Image.open(path).convert("RGB"))
        out = self.transform(image=img)
        return out["image"], label


def build_train_transform() -> A.Compose:
    """Augmentation pipeline applied to the TRAIN set only.
    CLAHE here mirrors the parameters used in inference (clipLimit=2.0,
    tileGrid=(8,8)) so the model learns to handle CLAHE'd inputs natively."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.ElasticTransform(alpha=20, sigma=4, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform() -> A.Compose:
    """Validation / test pipeline — deterministic, no augmentation."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_weighted_sampler(dataset: TumorDataset) -> WeightedRandomSampler:
    """Class-balanced sampler — every class gets equal expected pulls per epoch.
    Counters the meningioma underrepresentation in the combined pool (2 821
    images vs notumor's 4 121)."""
    counts = np.zeros(NUM_CLS, dtype=np.int64)
    for _, y in dataset.samples:
        counts[y] += 1
    class_weights = 1.0 / counts.astype(np.float64)
    sample_weights = [class_weights[y] for _, y in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


# ── Model building ─────────────────────────────────────────────────
def build_model(name: str, num_classes: int = NUM_CLS) -> nn.Module:
    cfg = MODEL_CONFIGS[name]
    model = cfg["factory"](weights=cfg["weights"])
    # Replace the classifier head with a Linear(num_classes) of the matching in_features.
    if name == "convnext_tiny":
        # convnext.classifier = Sequential(LayerNorm, Flatten, Linear)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    elif name == "efficientnet_b3":
        # efficientnet.classifier = Sequential(Dropout, Linear)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    elif name == "resnet50":
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def set_requires_grad(model: nn.Module, head_attr: str, frozen_backbone: bool) -> None:
    """Freeze/unfreeze everything except the classification head."""
    head = getattr(model, head_attr)
    for p in model.parameters():
        p.requires_grad = False
    if frozen_backbone:
        for p in head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True


# ── Mixup ──────────────────────────────────────────────────────────
def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Single-batch mixup. Returns (x_mixed, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1.0 - lam) * x[idx]
    return x_mixed, y, y[idx], float(lam)


# ── Train / eval loops ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion, device, use_mixup):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            if use_mixup and np.random.rand() < 0.3:
                xm, ya, yb, lam = mixup(x, y, alpha=0.2)
                logits = model(xm)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                logits = model(x)
                loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


# ── Single-model training driver ───────────────────────────────────
def train_one_model(name: str, epochs: int, batch_size: int, num_workers: int,
                    device: torch.device) -> dict:
    print(f"\n{'=' * 70}\n  TRAINING: {name}\n{'=' * 70}", flush=True)
    cfg = MODEL_CONFIGS[name]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Data
    train_ds = TumorDataset(SPLIT_DIR / "train", build_train_transform())
    val_ds   = TumorDataset(SPLIT_DIR / "val",   build_eval_transform())
    print(f"  train={len(train_ds)}  val={len(val_ds)}", flush=True)
    sampler = build_weighted_sampler(train_ds)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)

    # Model
    model = build_model(name).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict] = []

    # ── Phase 1: head only, 5 epochs ────────────────────────────
    set_requires_grad(model, cfg["head_attr"], frozen_backbone=True)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optim_head = torch.optim.AdamW(head_params, lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
    print("  [Phase 1] frozen backbone, head warmup", flush=True)
    for ep in range(5):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optim_head, scaler, criterion,
                                      device, use_mixup=False)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        row = {"phase": 1, "epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": vl_loss, "val_acc": vl_acc, "secs": round(dt, 1)}
        history.append(row)
        print(f"    P1 ep{ep + 1}/5  train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  ({dt:.1f}s)", flush=True)

    # ── Phase 2: unfreeze, cosine LR ─────────────────────────────
    set_requires_grad(model, cfg["head_attr"], frozen_backbone=False)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr_full"],
                              weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    best_val_acc = 0.0
    best_path = OUT_DIR / f"{name}.pth"
    print(f"  [Phase 2] full fine-tune, cosine LR, {epochs} epochs", flush=True)
    patience, no_improve = 8, 0
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, scaler, criterion,
                                      device, use_mixup=True)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        sched.step()
        dt = time.time() - t0
        row = {"phase": 2, "epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": vl_loss, "val_acc": vl_acc, "secs": round(dt, 1),
               "lr": sched.get_last_lr()[0]}
        history.append(row)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_path)
            star = "*"
            no_improve = 0
        else:
            star = " "
            no_improve += 1
        print(f"    P2 ep{ep + 1}/{epochs}{star} train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}  ({dt:.1f}s)", flush=True)
        if no_improve >= patience:
            print(f"    early stop — no val_acc improvement in {patience} epochs", flush=True)
            break

    # Save history
    hist_path = REPORTS / f"train_v2_{name}_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "epoch", "train_loss", "train_acc",
                                          "val_loss", "val_acc", "secs", "lr"])
        w.writeheader()
        for row in history:
            row.setdefault("lr", "")
            w.writerow(row)

    print(f"\n  best val_acc = {best_val_acc:.4f}  ->  {best_path}", flush=True)
    return {"model": name, "best_val_acc": best_val_acc, "checkpoint": str(best_path)}


# ── Entry point ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS) + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU    = {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  CUDA   = {torch.version.cuda}", flush=True)
    print(f"  train dir = {SPLIT_DIR / 'train'}", flush=True)
    print(f"  output    = {OUT_DIR}", flush=True)

    targets = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    summary = []
    overall_t0 = time.time()
    for name in targets:
        res = train_one_model(name, args.epochs, args.batch, args.workers, device)
        summary.append(res)
    overall_dt = time.time() - overall_t0

    print("\n" + "=" * 70 + "\n  SUMMARY\n" + "=" * 70)
    for r in summary:
        print(f"  {r['model']:18s}  best_val_acc={r['best_val_acc']:.4f}  -> {r['checkpoint']}")
    print(f"\n  total wall-clock: {overall_dt / 60:.1f} min")


if __name__ == "__main__":
    main()
