# Brain Tumor Detection Pipeline with LLM Integration

Multimodal brain MRI classification system with explainability, uncertainty
quantification, adversarial robustness testing, and an LLM-generated
diagnostic report. Given a brain MRI slice, it returns not just a label but
the reasoning behind it — model agreement, uncertainty, visual explanations,
and a plain-language report from a medical vision-language model.

> **Research / educational project.** Not a medical device. Not validated by
> any regulatory authority. Not for clinical decision-making.

---

## Demo

Don't want to run the whole stack? [`notebooks/demo.ipynb`](notebooks/demo.ipynb)
walks through the full pipeline end-to-end on a real MRI — classification,
uncertainty, explainability heat-maps, tumor segmentation, robustness testing,
and a real MedGemma-generated report — with every output already captured, so
it renders fully on GitHub with no setup required.

## Results at a glance

3-model ensemble evaluated on a locked, deduplicated test set of 2,114 MRI
scans (never seen during training):

| Model | Test accuracy | Balanced accuracy |
|---|---:|---:|
| ConvNeXt-Tiny | 99.29% | 99.25% |
| EfficientNet-B3 | 99.48% | 99.48% |
| ResNet-50 | 99.34% | 99.30% |
| **Ensemble (soft-vote)** | **99.48%** | **99.48%** |

Full per-class precision/recall/F1, confusion matrices, and latency numbers
are in [`reports/`](reports/) — see `reports/v2_metrics.json` for the raw
numbers used to regenerate the app's Metrics page.

---

## 1. What the system does

Given a brain MRI image (axial T1), the pipeline returns:

| Output | How it's produced |
|---|---|
| **Tumor type** — `glioma / meningioma / pituitary / no_tumor` | Ensemble of 3 CNNs (ConvNeXt-Tiny + EfficientNet-B3 + ResNet-50), each with test-time augmentation |
| **Confidence + uncertainty** | MC Dropout (20 stochastic passes) + epistemic/aleatoric decomposition |
| **Out-of-distribution check** | Energy-based score against the training distribution |
| **Focus-crop consistency check** | Re-classifies the model's own Grad-CAM++ attention region; disagreement flags unreliable reasoning |
| **Visual explanations** | 4-level hierarchical XAI: Grad-CAM, Grad-CAM++, LayerCAM (block4), fused LayerCAM |
| **Tumor bounding box + size** | YOLO (Cheng dataset) + MobileSAM segmentation, cross-validated against an independent MedGemma vision assessment |
| **Adversarial robustness** | Re-classifies under blur / brightness / noise / scanner-artifact perturbation; flags fragile predictions |
| **Malignancy score (0–10)** | Type-based clinical baseline × confidence × size-derived bonus |
| **Anatomical localization** | Region label mapped to a 3D brain atlas |
| **Diagnostic report** | MedGemma 1.5 4B (via Ollama) — basic / advanced modes, EN / ES |
| **Conversational Q&A** | Chat panel with patient/doctor audiences, scan-context-aware |

---

## 2. Folder layout

```
Tumor-detection/
├── README.md                      ← this file
├── notebooks/
│   └── demo.ipynb                 ← executed walkthrough of the full pipeline
├── app/                           ← runnable application
│   ├── app.py                     ← Flask backend (HTTP server on :7860)
│   ├── requirements.txt           ← Python dependencies
│   ├── cheng_yolo.pt              ← YOLO weights for tumor bbox (not tracked in git — see below)
│   ├── llm/                       ← MedGemma client + Ollama prompts
│   ├── preprocessing/              ← brain extraction, CLAHE, denoising
│   ├── models/                    ← trained weights (not tracked in git — see below)
│   │   └── v2/                    ← CNN ensemble (ConvNeXt + EffNet + ResNet)
│   ├── static/                    ← built React UI + 3D brain atlas GLBs
│   ├── frontend/                  ← React source (npm run build → static/)
│   └── src/                       ← training + evaluation scripts
│       ├── train_v2.py            ← train any of the 3 backbones
│       └── eval_v2.py             ← test-set metrics + confusion matrices
├── dataset/                       ← combined, deduplicated split (not tracked in git)
│   ├── train/  val/  test/
├── docs/                          ← figures for the paper / presentation
└── reports/                       ← evaluation metrics + confusion matrices (tracked)
```

**Not tracked in git**: `dataset/` (~280 MB), `app/models/` (~430 MB — individual
checkpoints exceed GitHub's 100 MB file limit), `app/frontend/node_modules/`,
`app/.venv/`. These are excluded via `.gitignore` to keep the repo a browsable
portfolio piece rather than an asset dump — see the demo notebook or
`reports/` for pre-computed results, or the setup steps below to run it
yourself with your own weights/dataset.

---

## 3. Requirements

| Component | Version | Why |
|---|---|---|
| **Python** | 3.11 (torch's CPU wheel doesn't yet support 3.12+) | Backend |
| **Node.js** | 18+ | Frontend (only if rebuilding UI) |
| **Ollama** | latest | Serves MedGemma 1.5 4B locally |
| **GPU** | Optional — NVIDIA recommended for training | Inference runs fine on CPU (`torch==2.2.0+cpu`); MedGemma calls are slower without a GPU (a few minutes per report on CPU vs. seconds on GPU) |
| **OS** | Windows 10/11, Linux | Tested on both Windows 11 (CPU-only) and Linux + RTX GPU |

---

## 4. First-time setup

### 4.1 Install Python dependencies

```powershell
cd app
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 Pull MedGemma into Ollama

```powershell
# Install Ollama first: https://ollama.com/download
ollama pull medgemma1.5:4b
```

To verify: `ollama list` should show `medgemma1.5:4b`.

### 4.3 Provide model weights and dataset

Not included in this repo (see §2). Either train your own (§6) or place
existing checkpoints at `app/models/v2/{convnext_tiny,efficientnet_b3,resnet50}.pth`
and a MobileSAM/YOLO checkpoint at `app/models/mobile_sam.pt` / `app/cheng_yolo.pt`.

### 4.4 (Optional) Rebuild the frontend

The `static/` folder already contains the built UI. Only rebuild if you modify React source:

```powershell
cd app\frontend
npm install
npm run build
# then copy build output back into ../static/
xcopy /E /I /Y build\* ..\static\
```

---

## 5. Running the app

```powershell
cd app
py app.py
```

You should see:

```
[BOOT] Importing libraries...
[BOOT] All imports OK
Pre-loading models...
[get_models] loading v2 weights from models/v2/
[get_models] v2 ensemble loaded (3 models)
Starting server on port 7860...
 * Running on http://127.0.0.1:7860
```

Open **http://127.0.0.1:7860** in a browser.

### Using the UI

1. Drag-and-drop or click to upload an MRI image (PNG / JPG)
2. Click **Analyze MRI**
3. Results panel shows:
   - Tumor type + confidence, per-model agreement
   - Uncertainty / OOD / Focus-Crop self-check
   - Malignancy assessment with bbox + size (pipeline estimate + independent MedGemma estimate)
   - Hierarchical Grad-CAM/LayerCAM visual explanations
   - Adversarial robustness score
   - MedGemma diagnostic report (Basic / Advanced toggle, EN / ES)
   - Brain atlas region link
4. **Floating chat (bottom-right)** — ask follow-up questions to MedGemma about the scan; switches between Patient and Doctor audience

### Language

Toggle EN ↔ ES with the language switcher in the header. Reports, chat, UI labels, and clinical text all translate.

---

## 6. Retraining the CNN ensemble

```powershell
cd app
py src\train_v2.py --model convnext_tiny  --epochs 40 --batch 32
py src\train_v2.py --model efficientnet_b3 --epochs 40 --batch 32
py src\train_v2.py --model resnet50       --epochs 40 --batch 32
```

This reads from `../dataset/train` and `../dataset/val`, applies the augmentation pipeline (HorizontalFlip, Rotate ±15°, BrightnessContrast, CLAHE, GaussNoise, ElasticTransform — all online), and saves checkpoints under `models/v2/`.

### Evaluating

```powershell
py src\eval_v2.py
```

Prints test-set accuracy, per-class precision/recall/F1, and writes a confusion matrix + `reports/v2_metrics.json` (the exact numbers the app's Metrics page reads).

---

## 7. Dataset

Combined from three publicly available brain MRI datasets, MD5-deduplicated
against each other (4 classes, T1-weighted axial slices):

| Split | Count |
|---|---:|
| Train | 9,867 |
| Val | 2,114 |
| Test | 2,114 |
| **Total** | **14,095** |

Per class (train):

| Class | Count |
|---|---:|
| Glioma | 2,808 |
| Meningioma | 1,975 |
| No Tumor | 2,885 |
| Pituitary | 2,199 |

Augmentation generates ≈ 380,866 augmented views during 40-epoch training. See `docs/augmentation_overview_es.png` for the full breakdown.

---

## 8. Architecture summary

```
                              MRI input (PNG / JPG)
                                       │
                                       ▼
                       ┌──────── Preprocessing ────────┐
                       │  Brain extraction · CLAHE     │
                       │  Sequence detector (T1 vs T2) │
                       └────────────────┬──────────────┘
                                        │
                  ┌─────────────────────┼─────────────────────┐
                  ▼                     ▼                     ▼
           CNN ensemble (×3)     Hierarchical XAI       YOLO + MobileSAM
         ConvNeXt + EffNet+ResN  Grad-CAM → LayerCAM     (tumor bbox + mask)
                  │                     │                     │
                  ▼                     ▼                     ▼
            MC Dropout            Focus-crop            Bounding box
          + epistemic σ          consistency check       + size %
          + OOD energy score     + robustness test
                                        │
                                        ▼
                              ┌─── MedGemma 1.5 4B ───┐
                              │  Vision cross-check    │
                              │  Diagnostic report      │
                              │  Conversational Q&A    │
                              └────────────┬───────────┘
                                           │
                                           ▼
                                   React UI (Material-UI)
                                   + 3D Brain Atlas (Three.js)
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not reach Ollama at http://127.0.0.1:11434` | Run `ollama serve` in another terminal, or check Ollama is installed |
| MedGemma report/assessment fails or times out | MedGemma 1.5 sometimes emits chain-of-thought instead of the requested format, especially on CPU. `llm/medgemma_client.py` has `_strip_thinking` (reports) and `_parse_assessment_json` (tumor size/location) to recover from this; click **Regenerate** to retry |
| MedGemma calls are very slow (minutes) | Expected on CPU-only inference — a 4B vision-language model has no GPU acceleration here. Budget a few minutes per report/assessment; timeouts are set generously (300–420s) to accommodate this |
| `CUDA out of memory` | Lower `--batch` (try 16 or 8); close other GPU apps |
| Port 7860 already in use | Kill the old process: `Get-NetTCPConnection -LocalPort 7860 \| Stop-Process -Id $_.OwningProcess -Force` |
| Frontend shows old version | Hard-refresh with `Ctrl + Shift + R` or clear cache |
| "no estimate" in MedGemma size card | MedGemma returned non-JSON text; check the Flask console for `[MedGemma assess] parse status=...` — `regex` means some fields were recovered, `failed` means none were |

---

## 10. Citation / credits

- **CNN backbones**: ConvNeXt-Tiny, EfficientNet-B3, ResNet-50 (ImageNet pretrained, fine-tuned)
- **MobileSAM**: Faster Segment Anything (https://github.com/ChaoningZhang/MobileSAM)
- **YOLO**: Ultralytics — Cheng dataset weights for tumor bbox
- **MedGemma 1.5 4B**: Google, January 2026 — medical VLM via Ollama
- **3D Brain Atlas**: 3D models served from `app/static/models/*.glb`

---

## License

Research / educational use only. Not for clinical decisions. The system is an academic project (TFG) and has not been validated by any regulatory authority (FDA, EMA, etc.).
