"""
Evaluate the 3 v2 models on the locked test set.

Produces:
    reports/v2_metrics.json            structured per-model + ensemble metrics
    reports/v2_confusion_<model>.png   confusion matrix figure (4x4 + counts)
    reports/v2_per_class.csv           per-class precision/recall/F1 table

Numbers reported per model:
    test_accuracy
    test_balanced_accuracy
    per-class precision / recall / F1
    ECE (Expected Calibration Error, 15 bins)
    latency_ms_single  / latency_ms_tta5
    param_count_M / model_size_mb

Ensemble:
    soft_vote_test_acc        — mean of softmax probs across the 3 models
    pairwise_agreement_pct    — how often each model pair agrees on the top-1
    three_way_agreement_pct   — how often all 3 agree

Run:
    python src/eval_v2.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

from train_v2 import (
    CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_CONFIGS,
    OUT_DIR, SPLIT_DIR, build_model,
)

REPORTS = Path(__file__).resolve().parents[2] / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


class _TestDataset(Dataset):
    def __init__(self, root: Path):
        self.tf = Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            cdir = root / cls
            for p in sorted(cdir.iterdir()):
                if p.is_file():
                    self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = np.array(Image.open(path).convert("RGB"))
        return self.tf(image=img)["image"], label


@torch.no_grad()
def collect_softmax(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error on the predicted-class confidence."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def measure_latency(model, device, n_warmup=10, n_runs=30) -> tuple[float, float]:
    """Return (single-image ms, TTA-5 ms) — wall-clock per image."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.time() - t0) / n_runs * 1000.0

    # TTA-5: original + hflip + 3 rotations
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            probs = F.softmax(model(dummy), dim=1)
            probs += F.softmax(model(torch.flip(dummy, dims=[3])), dim=1)
            for _r in range(3):
                probs += F.softmax(model(torch.rot90(dummy, k=1, dims=[2, 3])), dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    tta_ms = (time.time() - t0) / n_runs * 1000.0
    return single_ms, tta_ms


def save_confusion(cm: np.ndarray, name: str, acc: float) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=140)
    im = ax.imshow(cm, cmap="Purples", interpolation="nearest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=11, color=color, weight="bold")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}  ·  acc={acc:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = REPORTS / f"v2_confusion_{name}.png"
    plt.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}", flush=True)

    test_ds = _TestDataset(SPLIT_DIR / "test")
    print(f"  test set = {len(test_ds)}", flush=True)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")

    per_class_rows: list[dict] = []
    metrics: dict = {"models": [], "ensemble": {}, "classes": list(CLASSES)}
    all_probs = {}        # name -> (N, 4)
    labels_global: np.ndarray | None = None

    for name in ("convnext_tiny", "efficientnet_b3", "resnet50"):
        print(f"\n=== {name} ===", flush=True)
        ckpt = OUT_DIR / f"{name}.pth"
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        probs, labels = collect_softmax(model, loader, device)
        all_probs[name] = probs
        labels_global = labels  # same order every time
        preds = probs.argmax(axis=1)

        acc = float((preds == labels).mean())
        bal = float(balanced_accuracy_score(labels, preds))
        ece = ece_score(probs, labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, labels=list(range(len(CLASSES))), zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
        single_ms, tta_ms = measure_latency(model, device)
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = ckpt.stat().st_size / (1024 * 1024)

        print(f"  test_acc          = {acc:.4f}")
        print(f"  balanced_acc      = {bal:.4f}")
        print(f"  ECE (15 bins)     = {ece:.4f}")
        print(f"  latency single ms = {single_ms:.1f}")
        print(f"  latency TTA-5  ms = {tta_ms:.1f}")
        print(f"  params            = {param_count/1e6:.1f}M")
        print(f"  size              = {size_mb:.1f} MB")

        cm_png = save_confusion(cm, name, acc)
        print(f"  confusion -> {cm_png}")

        per_class = []
        for ci, cls in enumerate(CLASSES):
            per_class.append({
                "class": cls,
                "precision": float(precision[ci]),
                "recall":    float(recall[ci]),
                "f1":        float(f1[ci]),
                "support":   int(support[ci]),
            })
            per_class_rows.append({"model": name, **per_class[-1]})

        metrics["models"].append({
            "name":                    name,
            "test_accuracy":           acc,
            "test_balanced_accuracy":  bal,
            "ece":                     ece,
            "latency_ms_single":       round(single_ms, 2),
            "latency_ms_tta5":         round(tta_ms, 2),
            "param_count_M":           round(param_count / 1e6, 2),
            "model_size_mb":           round(size_mb, 1),
            "per_class":               per_class,
            "confusion_matrix":        cm.tolist(),
            "confusion_png":           cm_png.name,
        })

    # ── Ensemble ─────────────────────────────────────────────────
    if all_probs and labels_global is not None:
        # Soft-vote: mean of softmax probs
        mean_probs = np.mean(list(all_probs.values()), axis=0)
        ens_preds = mean_probs.argmax(axis=1)
        ens_acc   = float((ens_preds == labels_global).mean())
        ens_bal   = float(balanced_accuracy_score(labels_global, ens_preds))
        ens_cm    = confusion_matrix(
            labels_global, ens_preds, labels=list(range(len(CLASSES)))
        )

        names = list(all_probs.keys())
        preds_per_model = {n: all_probs[n].argmax(axis=1) for n in names}
        pairwise = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                agree = float((preds_per_model[a] == preds_per_model[b]).mean())
                pairwise[f"{a}__{b}"] = round(agree * 100, 2)
        three_way = float((
            (preds_per_model[names[0]] == preds_per_model[names[1]]) &
            (preds_per_model[names[1]] == preds_per_model[names[2]])
        ).mean()) * 100

        # How many test images the ensemble GOT RIGHT that at least one model
        # got wrong — i.e. where voting actually changed the outcome.
        ens_corrected = 0
        ens_outvoted  = 0
        for i in range(len(labels_global)):
            y = labels_global[i]
            per = [preds_per_model[n][i] for n in names]
            wrong_models = sum(1 for p in per if p != y)
            ens_right = (ens_preds[i] == y)
            if ens_right and wrong_models > 0:
                ens_corrected += 1
            elif (not ens_right) and any(p == y for p in per):
                ens_outvoted += 1

        print("\n=== ensemble (soft vote) ===")
        print(f"  test_acc                 = {ens_acc:.4f}")
        print(f"  balanced_acc             = {ens_bal:.4f}")
        for k, v in pairwise.items():
            print(f"  agree {k:50s} = {v:.2f}%")
        print(f"  three-way agreement      = {three_way:.2f}%")
        print(f"  ensemble corrected       = {ens_corrected} cases "
              f"(voting fixed >=1 model's error)")
        print(f"  ensemble outvoted        = {ens_outvoted} cases "
              f"(voting overrode a correct model)")

        ens_cm_png = save_confusion(ens_cm, "ensemble_soft_vote", ens_acc)
        print(f"  confusion -> {ens_cm_png}")

        metrics["ensemble"] = {
            "soft_vote_test_acc":      ens_acc,
            "soft_vote_balanced_acc":  ens_bal,
            "pairwise_agreement_pct":  pairwise,
            "three_way_agreement_pct": round(three_way, 2),
            "confusion_matrix":        ens_cm.tolist(),
            "confusion_png":           ens_cm_png.name,
            "ensemble_corrected_cases":   ens_corrected,
            "ensemble_outvoted_cases":    ens_outvoted,
        }

    # ── Persist ───────────────────────────────────────────────────
    out_json = REPORTS / "v2_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")

    out_csv = REPORTS / "v2_per_class.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "class", "precision", "recall", "f1", "support"])
        w.writeheader()
        for r in per_class_rows:
            w.writerow(r)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
