"""Test 4 ensemble voting strategies on the locked test set and compare.

For each model:
  - Collect raw logits on val (for temperature calibration) and test
For each strategy, build ensemble predictions on test, then report:
  - Accuracy, balanced accuracy
  - Confusion matrix
  - Number of cells differing from the current soft-vote matrix
  - Number of cells differing from EfficientNet-B3's matrix
  - Per-class errors

Strategies:
  1. Current   — softmax averaging (`mean(softmax(logits_i))`)
  2. Logit     — average logits then softmax (`softmax(mean(logits_i))`)
  3. T-soft    — per-model temperature-calibrated softmax averaging
  4. HardVote  — argmax of each model votes, majority wins, mean-conf tiebreak
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from train_v2 import (
    CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    OUT_DIR, SPLIT_DIR, build_model,
)


class _Dataset(Dataset):
    def __init__(self, root: Path):
        self.tf = Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.samples = []
        for idx, cls in enumerate(CLASSES):
            for p in sorted((root / cls).iterdir()):
                if p.is_file():
                    self.samples.append((p, idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.tf(image=np.array(Image.open(path).convert("RGB")))["image"], y


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def fit_temperature(logits_val: np.ndarray, labels_val: np.ndarray,
                    max_iter: int = 50) -> float:
    """Fit a single scalar temperature on validation NLL via LBFGS.
    Returns T such that softmax(logits / T) is the calibrated distribution."""
    L = torch.tensor(logits_val, dtype=torch.float32)
    y = torch.tensor(labels_val, dtype=torch.long)
    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / T.clamp(min=1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def summarize(name: str, preds: np.ndarray, labels: np.ndarray,
              soft_vote_preds: np.ndarray, effnet_preds: np.ndarray) -> dict:
    acc = float((preds == labels).mean())
    bal = float(balanced_accuracy_score(labels, preds))
    cm  = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    n_diff_soft  = int((preds != soft_vote_preds).sum())
    n_diff_eff   = int((preds != effnet_preds).sum())
    # Wins / losses vs each strategy
    soft_correct  = (soft_vote_preds == labels)
    eff_correct   = (effnet_preds == labels)
    self_correct  = (preds == labels)
    return {
        "name":          name,
        "acc":           acc,
        "bal_acc":       bal,
        "errors":        int(cm.sum() - np.trace(cm)),
        "cm":            cm,
        "diff_soft":     n_diff_soft,
        "diff_effnet":   n_diff_eff,
        "fixed_vs_soft":  int(((~soft_correct) & self_correct).sum()),
        "broke_vs_soft":  int((soft_correct & (~self_correct)).sum()),
        "fixed_vs_effnet": int(((~eff_correct) & self_correct).sum()),
        "broke_vs_effnet": int((eff_correct & (~self_correct)).sum()),
    }


def print_row(r: dict) -> None:
    print(f"  {r['name']:18s} acc={r['acc']*100:6.3f}%  bal={r['bal_acc']*100:6.3f}%  "
          f"err={r['errors']:>3d}  diff_soft={r['diff_soft']:>3d}  "
          f"diff_eff={r['diff_effnet']:>3d}  "
          f"vs_soft +{r['fixed_vs_soft']}/-{r['broke_vs_soft']}  "
          f"vs_eff +{r['fixed_vs_effnet']}/-{r['broke_vs_effnet']}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}")

    val_ds  = _Dataset(SPLIT_DIR / "val")
    test_ds = _Dataset(SPLIT_DIR / "test")
    print(f"  val = {len(val_ds)}  test = {len(test_ds)}")

    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=device.type == "cuda")

    val_logits, test_logits, test_softmax = {}, {}, {}
    labels_test = None
    for name in ("convnext_tiny", "efficientnet_b3", "resnet50"):
        print(f"\n[{name}] forward pass...")
        t0 = time.time()
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(OUT_DIR / f"{name}.pth", map_location=device))
        vl, yv = collect_logits(model, val_loader, device)
        tl, yt = collect_logits(model, test_loader, device)
        labels_test = yt
        val_logits[name]    = vl
        test_logits[name]   = tl
        test_softmax[name]  = softmax_np(tl)
        # Per-model accuracy as sanity
        preds = tl.argmax(axis=1)
        acc   = float((preds == yt).mean())
        print(f"  test_acc        = {acc*100:.3f}%  ({time.time()-t0:.1f}s)")
        del model

    # ── Temperature scaling on validation ─────────────────────────
    temperatures = {n: fit_temperature(val_logits[n], yv) for n in val_logits}
    print("\nFitted temperatures (val NLL):")
    for n, T in temperatures.items():
        print(f"  {n:18s} T = {T:.3f}")

    # ── Build ensemble predictions per strategy ───────────────────
    names = list(test_logits.keys())
    # 1. Current (softmax averaging)
    mean_softmax  = np.mean([test_softmax[n] for n in names], axis=0)
    soft_preds    = mean_softmax.argmax(axis=1)
    # 2. Logit averaging
    mean_logits   = np.mean([test_logits[n] for n in names], axis=0)
    logit_preds   = softmax_np(mean_logits).argmax(axis=1)
    # 3. Temperature-calibrated soft
    tsoft_probs   = np.mean(
        [softmax_np(test_logits[n] / temperatures[n]) for n in names], axis=0
    )
    tsoft_preds   = tsoft_probs.argmax(axis=1)
    # 4. Hard majority vote with mean-confidence tiebreak
    votes         = np.stack([test_softmax[n].argmax(axis=1) for n in names])  # (3, N)
    hard_preds    = np.zeros_like(labels_test)
    for i in range(len(labels_test)):
        v = votes[:, i]
        # bincount over 4 classes
        c = np.bincount(v, minlength=len(CLASSES))
        winners = np.where(c == c.max())[0]
        if len(winners) == 1:
            hard_preds[i] = int(winners[0])
        else:
            # Tie: use mean softmax confidence on each winner class
            scores = mean_softmax[i, winners]
            hard_preds[i] = int(winners[scores.argmax()])

    effnet_preds = test_logits["efficientnet_b3"].argmax(axis=1)

    print("\n=== Strategy comparison (test set, n=2114) ===")
    print(f"  {'strategy':18s} {'acc':>10s} {'bal':>10s} {'err':>5s} "
          f"{'diff_soft':>10s} {'diff_eff':>9s} {'vs_soft':>14s} {'vs_eff':>14s}")
    rows = []
    for nm, preds in [
        ("CurrentSoft",       soft_preds),
        ("LogitAvg",          logit_preds),
        ("T-CalibSoft",       tsoft_preds),
        ("HardMajority",      hard_preds),
    ]:
        r = summarize(nm, preds, labels_test, soft_preds, effnet_preds)
        rows.append(r)
        print_row(r)

    # Per-model baseline reminders
    print("\n  Per-model test accuracy (for reference):")
    for n in names:
        preds = test_logits[n].argmax(axis=1)
        acc = float((preds == labels_test).mean())
        print(f"    {n:18s} {acc*100:6.3f}%  (err={(preds!=labels_test).sum()})")

    # Show every confusion matrix
    for r in rows:
        print(f"\n--- confusion: {r['name']} (acc {r['acc']*100:.3f}%) ---")
        for i, c in enumerate(CLASSES):
            row = "  ".join(f"{r['cm'][i, j]:>4d}" for j in range(len(CLASSES)))
            print(f"  true_{c:11s} {row}")

    # Winner pick
    winner = max(rows, key=lambda r: r["acc"])
    print(f"\n=== BEST: {winner['name']} at {winner['acc']*100:.3f}% "
          f"({winner['errors']} errors / 2114) ===")


if __name__ == "__main__":
    main()
