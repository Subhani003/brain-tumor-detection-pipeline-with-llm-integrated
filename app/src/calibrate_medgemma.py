"""Calibration study — measure MedGemma's standalone class-prediction accuracy
on a stratified sample of the locked v2 test set.

Sampling plan (seed=42):
  * ALL `disagreement` cases  — where the three v2 CNNs do not agree on top-1.
    These are the cases where a tiebreaker matters.
  * 50 `agreement` cases       — 3-way CNN agreement, balanced across the 4
    classes. Tells us whether MedGemma is even right on the easy stuff.
  * remaining random fill      — to reach a 150-image sample so the per-class
    P/R/F1 numbers carry some signal.

For each image:
  1. Resize to 512 (preserve aspect ratio).
  2. POST to Ollama with a focused single-class classification prompt
     (no free text, JSON-only response).
  3. Parse the returned class.
  4. Compare to ground truth.

Outputs:
  reports/medgemma_calibration.json   per-image results + per-stratum stats
"""
from __future__ import annotations

import base64
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_v2 import (
    CLASSES, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    OUT_DIR, SPLIT_DIR, build_model,
)

# Reuse the Ollama client transport directly so we can use a custom prompt.
import urllib.error
import urllib.request

ROOT       = Path(__file__).resolve().parents[2]
REPORTS    = ROOT / "reports"
OUT_FILE   = REPORTS / "medgemma_calibration.json"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL      = "medgemma1.5:4b"
SEED       = 42
SAMPLE_SIZE = 150

# Maps anything MedGemma might say to one of our 4 canonical class indices.
# Index order MUST match CLASSES = ("glioma", "meningioma", "notumor", "pituitary").
_CLASS_ALIASES = {
    "glioma":     0,
    "meningioma": 1,
    "notumor":    2,
    "no_tumor":   2,
    "no tumor":   2,
    "none":       2,
    "normal":     2,
    "healthy":    2,
    "pituitary":  3,
    "pituitary tumor": 3,
    "pituitary adenoma": 3,
}


CLASSIFY_PROMPT = (
    "You are a neuroradiology AI. Look at this brain MRI axial slice and "
    "classify it into EXACTLY ONE of these 4 classes:\n"
    "  glioma     - infiltrative tumor of glial cells, usually intra-axial\n"
    "  meningioma - extra-axial mass arising from the meninges, often with dural attachment\n"
    "  notumor    - normal brain, no tumor visible\n"
    "  pituitary  - mass in the sella turcica at the base of the brain\n\n"
    "Respond with ONLY a single JSON object, no other text, no markdown fences:\n"
    '{"class": "glioma|meningioma|notumor|pituitary"}\n'
)


# ── Step 1: collect CNN softmax on test set, identify strata ─────────
class _TestDataset(Dataset):
    def __init__(self, root: Path):
        self.tf = Compose([
            Resize(IMG_SIZE, IMG_SIZE),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(CLASSES):
            for p in sorted((root / cls).iterdir()):
                if p.is_file():
                    self.samples.append((p, idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.tf(image=np.array(Image.open(path).convert("RGB")))["image"], y


@torch.no_grad()
def _collect_preds(model, loader, device):
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            preds.append(model(x).float().argmax(dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def build_strata(device):
    ds = _TestDataset(SPLIT_DIR / "test")
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")
    per_model: dict[str, np.ndarray] = {}
    labels: np.ndarray | None = None
    for name in ("convnext_tiny", "efficientnet_b3", "resnet50"):
        print(f"[strata] {name} forward...", flush=True)
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(OUT_DIR / f"{name}.pth", map_location=device))
        preds, ys = _collect_preds(model, loader, device)
        per_model[name] = preds
        labels = ys
        del model

    # Disagreement = any two model preds differ
    pred_stack = np.stack(list(per_model.values()))
    all_agree = (pred_stack == pred_stack[0]).all(axis=0)
    disagree_idx = np.where(~all_agree)[0].tolist()
    agree_idx    = np.where(all_agree)[0].tolist()
    print(f"[strata] test = {len(ds)}  agree = {len(agree_idx)}  disagree = {len(disagree_idx)}")

    rng = random.Random(SEED)

    # Bucket the agreement indices by ground-truth class for balanced sampling.
    by_class: dict[int, list[int]] = {c: [] for c in range(len(CLASSES))}
    for i in agree_idx:
        by_class[int(labels[i])].append(i)

    per_class_quota = 50 // len(CLASSES)
    confident_picks: list[int] = []
    for c in range(len(CLASSES)):
        pool = by_class[c]
        rng.shuffle(pool)
        confident_picks.extend(pool[:per_class_quota])

    fill_target = SAMPLE_SIZE - len(disagree_idx) - len(confident_picks)
    picked_set = set(disagree_idx) | set(confident_picks)
    remaining = [i for i in range(len(ds)) if i not in picked_set]
    rng.shuffle(remaining)
    random_picks = remaining[:max(0, fill_target)]

    final_idx = list(disagree_idx) + list(confident_picks) + list(random_picks)
    final_idx.sort()
    print(f"[strata] picked sample: {len(final_idx)} images  "
          f"(disagree={len(disagree_idx)}, confident={len(confident_picks)}, "
          f"random={len(random_picks)})")

    return {
        "samples":       [(str(ds.samples[i][0]), int(ds.samples[i][1])) for i in final_idx],
        "indices":       final_idx,
        "labels":        labels.tolist(),
        "all_agree":     all_agree.tolist(),
        "per_model":     {n: p.tolist() for n, p in per_model.items()},
        "disagree_idx":  disagree_idx,
        "agree_idx":     agree_idx,
    }


# ── Step 2: query MedGemma per image ────────────────────────────────
def _image_to_b64(path: str, max_side: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            img = img.resize((max_side, int(h * max_side / w)))
        else:
            img = img.resize((int(w * max_side / h), max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_class(raw: str) -> int | None:
    """Parse MedGemma's reply, return canonical class index or None."""
    text = (raw or "").strip()
    # Try JSON parsing first.
    for candidate in (text,) + tuple(_extract_json_objects(text)):
        try:
            obj = json.loads(candidate)
            cls_text = str(obj.get("class") or "").strip().lower()
            if cls_text in _CLASS_ALIASES:
                return _CLASS_ALIASES[cls_text]
        except (json.JSONDecodeError, AttributeError):
            continue
    # Fallback — look for any literal class word in the text.
    lower = text.lower()
    for alias, idx in _CLASS_ALIASES.items():
        if alias in lower:
            return idx
    return None


def _extract_json_objects(text: str):
    """Yield substrings that look like JSON object literals."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def query_medgemma(image_b64: str, timeout: int = 60) -> tuple[str, float]:
    body = {
        "model":    MODEL,
        "prompt":   CLASSIFY_PROMPT,
        "images":   [image_b64],
        "stream":   False,
        "options":  {
            "temperature":    0.0,
            "num_ctx":        2048,
            "num_predict":    60,
            "repeat_penalty": 1.1,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip(), time.time() - t0


# ── Driver ──────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"  device = {device}")

    print("\n[1/3] Building strata...")
    strata = build_strata(device)

    print("\n[2/3] Querying MedGemma on the picked images...")
    results: list[dict] = []
    n_ok, n_err = 0, 0
    n_picked = len(strata["samples"])
    for k, (path, gt) in enumerate(strata["samples"], 1):
        idx_in_test = strata["indices"][k - 1]
        is_disagree = idx_in_test in strata["disagree_idx"]
        try:
            b64 = _image_to_b64(path)
            raw, dt = query_medgemma(b64)
            pred = _parse_class(raw)
            ok = (pred == gt)
            n_ok += int(ok is True)
            n_err += int(pred is None)
            cnn_preds = {n: int(strata["per_model"][n][idx_in_test])
                         for n in strata["per_model"]}
            results.append({
                "idx":          idx_in_test,
                "path":         path,
                "ground_truth": int(gt),
                "ground_truth_name": CLASSES[gt],
                "medgemma_pred":      pred,
                "medgemma_pred_name": (CLASSES[pred] if pred is not None else None),
                "medgemma_raw":       raw[:200],
                "medgemma_latency_s": round(dt, 2),
                "correct":      bool(ok) if pred is not None else None,
                "is_disagreement_case": is_disagree,
                "cnn_preds":    {n: CLASSES[v] for n, v in cnn_preds.items()},
            })
            print(f"  [{k:>3d}/{n_picked}] {CLASSES[gt]:11s} -> MG={('' if pred is None else CLASSES[pred]):11s}  "
                  f"{'OK' if ok else ('?' if pred is None else 'X')}  "
                  f"({dt:.1f}s)  disagree={is_disagree}", flush=True)
        except (urllib.error.URLError, Exception) as e:
            print(f"  [{k:>3d}/{n_picked}] ERROR: {e}", flush=True)
            results.append({
                "idx": idx_in_test, "path": path, "ground_truth": int(gt),
                "ground_truth_name": CLASSES[gt], "error": str(e),
            })

    # ── Step 3: aggregate ────────────────────────────────────────────
    print("\n[3/3] Aggregating...")
    answered = [r for r in results if r.get("medgemma_pred") is not None]
    unparseable = [r for r in results if "error" not in r and r.get("medgemma_pred") is None]
    errors = [r for r in results if "error" in r]

    overall_acc = float(np.mean([r["correct"] for r in answered])) if answered else 0.0
    by_class_acc = {}
    for ci, cls in enumerate(CLASSES):
        rows = [r for r in answered if r["ground_truth"] == ci]
        if rows:
            by_class_acc[cls] = {
                "n":   len(rows),
                "acc": float(np.mean([r["correct"] for r in rows])),
            }

    disagree_rows = [r for r in answered if r["is_disagreement_case"]]
    agree_rows    = [r for r in answered if not r["is_disagreement_case"]]
    disagree_acc = float(np.mean([r["correct"] for r in disagree_rows])) if disagree_rows else None
    agree_acc    = float(np.mean([r["correct"] for r in agree_rows]))    if agree_rows    else None

    # Confusion matrix
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for r in answered:
        cm[r["ground_truth"]][r["medgemma_pred"]] += 1

    summary = {
        "model":              MODEL,
        "sample_size":        len(results),
        "answered":           len(answered),
        "unparseable":        len(unparseable),
        "errors":             len(errors),
        "overall_accuracy":   round(overall_acc, 4),
        "agreement_zone_acc": round(agree_acc, 4) if agree_acc is not None else None,
        "disagreement_zone_acc": round(disagree_acc, 4) if disagree_acc is not None else None,
        "per_class_accuracy": {k: {"n": v["n"], "acc": round(v["acc"], 4)}
                                for k, v in by_class_acc.items()},
        "confusion_matrix":   cm.tolist(),
    }

    OUT_FILE.write_text(json.dumps({"summary": summary, "per_image": results}, indent=2),
                        encoding="utf-8")
    print(f"\nWrote {OUT_FILE}")
    print()
    print("====================  SUMMARY  ====================")
    print(f"  sample size:           {summary['sample_size']}")
    print(f"  answered (parsed):     {summary['answered']}")
    print(f"  unparseable / errors:  {summary['unparseable']} / {summary['errors']}")
    print(f"  overall accuracy:      {summary['overall_accuracy']*100:6.2f}%")
    print(f"  agreement-zone acc:    "
          f"{(summary['agreement_zone_acc'] or 0)*100:6.2f}%  "
          f"(n={len(agree_rows)})")
    print(f"  disagreement-zone acc: "
          f"{(summary['disagreement_zone_acc'] or 0)*100:6.2f}%  "
          f"(n={len(disagree_rows)})")
    print("  per-class accuracy:")
    for cls, info in summary["per_class_accuracy"].items():
        print(f"    {cls:11s} n={info['n']:>3d}  acc={info['acc']*100:6.2f}%")
    print("  confusion (rows=truth, cols=MedGemma pred):")
    print("              " + " ".join(f"{c[:5]:>6s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        row = " ".join(f"{cm[i][j]:>6d}" for j in range(len(CLASSES)))
        print(f"  true_{c:9s} {row}")


if __name__ == "__main__":
    main()
