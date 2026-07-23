# Brain Tumor Detection — Proyecto v2 (clean)

Multimodal brain MRI classification system with explainability, uncertainty quantification, and an LLM-assisted diagnostic report. This is the clean, self-contained version of the project.

---

## 1. What the system does

Given a brain MRI image (axial T1), the pipeline returns:

| Output | How it's produced |
|---|---|
| **Tumor type** — `glioma / meningioma / pituitary / no_tumor` | Ensemble of 3 CNNs (ConvNeXt-Tiny + EfficientNet-B3 + ResNet-50) trained on 9,867 images |
| **Confidence + uncertainty** | MC Dropout + epistemic/aleatoric decomposition |
| **OOD detection** | Energy-based score against the training distribution |
| **Anomaly first-gate** | VQ-VAE reconstruction + autoregressive transformer NLL (optional toggle in the UI) |
| **Tumor bounding box + size** | YOLO (Cheng) + MobileSAM segmentation + MedGemma VLM cross-validation |
| **Malignancy score (0–10)** | Type-based clinical baseline × confidence × size-derived bonus |
| **Anatomical localization** | Region label mapped to a 3D brain atlas |
| **Diagnostic report** | MedGemma 1.5 4B Multimodal LLM via Ollama (basic / advanced modes, EN / ES) |
| **Conversational Q&A** | Chat panel with patient/doctor audiences, scan-context-aware |

---

## 2. Folder layout

```
Proyecto-Tumor-Clean/
├── README.md                      ← this file
├── app/                           ← runnable application
│   ├── app.py                     ← Flask backend (HTTP server on :7860)
│   ├── requirements.txt           ← Python dependencies
│   ├── cheng_yolo.pt              ← YOLO weights for tumor bbox (5 MB)
│   ├── mobile_sam.pt              ← MobileSAM segmentation (39 MB)
│   ├── llm/                       ← MedGemma client + Ollama prompts
│   ├── preprocessing/             ← brain extraction, CLAHE, denoising
│   ├── vqvae/                     ← VQ-VAE anomaly detector
│   ├── models/                    ← all trained weights (~615 MB)
│   │   ├── v2/                    ← v2 CNN ensemble (ConvNeXt + EffNet + ResNet)
│   │   ├── vqvae/                 ← VQ-VAE + LatentTransformer + thresholds
│   │   ├── *_best.pth             ← v1 legacy models (loaded as fallback)
│   │   ├── *.pkl / *.json         ← Nyul landmarks + OOD calibration
│   ├── static/                    ← built React UI + 3D brain atlas GLBs
│   ├── frontend/                  ← React source (npm run build → static/)
│   ├── src/                       ← training + evaluation scripts
│   │   ├── train_v2.py            ← train any of the 3 backbones
│   │   ├── eval_v2.py             ← test-set metrics + confusion matrix
│   │   ├── calibrate_medgemma.py  ← OOD threshold calibration
│   │   ├── compare_voting.py      ← ensemble voting strategy ablation
│   │   └── generate_methodology_doc.py
│   └── make_preprocessing_preview.py
├── dataset/                       ← BRISC2025 split (deduplicated, 70/15/15)
│   ├── train/                     ← 9,867 images (4 classes)
│   ├── val/                       ← 2,114 images
│   └── test/                      ← 2,114 images
└── docs/                          ← figures for the paper / presentation
    ├── augmentation_overview_es.png
    ├── augmentation_slide_es.pptx
    └── augmentation_slide_es_editable.pptx
```

---

## 3. Requirements

| Component | Version | Why |
|---|---|---|
| **Python** | 3.11+ | Backend |
| **Node.js** | 18+ | Frontend (only if rebuilding UI) |
| **Ollama** | latest | Serves MedGemma 1.5 4B locally |
| **GPU** | NVIDIA, ≥ 8 GB VRAM | CNN ensemble + VQ-VAE + MedGemma |
| **OS** | Windows 10 / 11, Linux | Tested on Win 11 with RTX 4060 Laptop |

---

## 4. First-time setup

### 4.1 Install Python dependencies

```powershell
cd app
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 Pull MedGemma into Ollama

```powershell
# Install Ollama first: https://ollama.com/download
ollama pull medgemma1.5:4b
```

To verify: `ollama list` should show `medgemma1.5:4b`.

### 4.3 (Optional) Rebuild the frontend

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
2. Optional: toggle the **VQ-VAE first-gate** switch above the Analyze button (simulates an anomaly detection stage before the CNN ensemble)
3. Click **Analyze MRI**
4. Results panel shows:
   - Tumor type + confidence
   - Malignancy assessment with bbox + size
   - Uncertainty / OOD / Focus-Crop self-check
   - MedGemma diagnostic report (Basic / Advanced toggle, EN / ES)
   - Brain atlas region link
5. **Floating chat (bottom-right)** — ask follow-up questions to MedGemma about the scan; switches between Patient and Doctor audience

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

Prints test-set accuracy, per-class precision/recall/F1, and writes a confusion matrix to `reports/`.

---

## 7. Dataset

Source: **BRISC2025** brain MRI tumor classification dataset (4 classes, T1-weighted axial slices).

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
           VQ-VAE gate          CNN ensemble (×3)       YOLO + MobileSAM
        (optional toggle)    ConvNeXt + EffNet + ResN  (tumor bbox + mask)
                  │                     │                     │
                  ▼                     ▼                     ▼
             Anomaly level       MC Dropout                Bounding box
             (LOW / MED / HI)    + epistemic σ             + size %
                                + OOD energy score
                                        │
                                        ▼
                              ┌─── MedGemma 1.5 4B ───┐
                              │  Tumor assessment      │
                              │  Diagnostic report     │
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
| `Empty reply from MedGemma` / strange text in chat | MedGemma 1.5 sometimes emits chain-of-thought. The backend has a `_strip_chat_thinking` and `_parse_assessment_json` filter — restart Flask if you updated the code |
| `CUDA out of memory` | Lower `--batch` (try 16 or 8); close other GPU apps |
| Port 7860 already in use | Kill the old process: `Get-NetTCPConnection -LocalPort 7860 | Stop-Process -Id $_.OwningProcess -Force` |
| Frontend shows old version | Hard-refresh with `Ctrl + Shift + R` or clear cache |
| "no estimate" in MedGemma size card | MedGemma returned non-JSON text; check Flask console for `[MedGemma assess] parse status=...` — `regex` or `failed` indicate model is misbehaving on that image |

---

## 10. Citation / credits

- **CNN backbones**: ConvNeXt-Tiny, EfficientNet-B3, ResNet-50 (ImageNet pretrained, fine-tuned)
- **VQ-VAE**: custom implementation in `app/vqvae/vqvae_model.py`
- **MobileSAM**: Faster Segment Anything (https://github.com/ChaoningZhang/MobileSAM)
- **YOLO**: Ultralytics — Cheng dataset weights for tumor bbox
- **MedGemma 1.5 4B**: Google, January 2026 — medical VLM via Ollama
- **3D Brain Atlas**: 3D models served from `app/static/models/*.glb`

---

## License

Research / educational use only. Not for clinical decisions. The system is an academic project (TFG) and has not been validated by any regulatory authority (FDA, EMA, etc.).
