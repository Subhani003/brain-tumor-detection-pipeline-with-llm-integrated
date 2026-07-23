"""
Brain Tumor Detection — Unified Flask app for Hugging Face Spaces.
Serves React frontend (static/) + prediction API (/api/predict).
"""
print("[BOOT] Importing libraries...", flush=True)
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import base64
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter
import cv2
import os
print("[BOOT] All imports OK", flush=True)

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
_models = {}
_device = None


# ── Model Architectures ────────────────────────────────────────────
class DenseNet169Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.densenet169(weights=None)
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.efficientnet_b3(weights=None)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)


class ResNet50Classifier(nn.Module):
    """Same head shape as the other deployed tumor classifiers."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.resnet50(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)


def _load_model(cls, filename):
    """Legacy v1 loader. Reads from models/{filename}, applies the v1 custom
    classification head wrapper. Used only when v2 weights are missing."""
    model_path = Path(__file__).parent / 'models' / filename
    model = cls(num_classes=4)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── v2 loader: stock torchvision model + last-layer Linear(4), wrapped so the
# existing get_target_layer / get_layercam_layers code that touches
# `model.base_model.features` keeps working unchanged.
class _V2Wrapper(nn.Module):
    """Thin wrapper that exposes the torchvision model as `self.base_model` so
    the Grad-CAM and LayerCAM layer pickers work without further changes."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base_model = base

    def forward(self, x):
        return self.base_model(x)


def _build_v2_torchvision(name: str) -> nn.Module:
    """Mirror of src/train_v2.build_model — same architectures, same head
    replacement. Importing from src/ at request time keeps app.py boot
    deterministic when v2 isn't present yet."""
    from torchvision.models import (
        convnext_tiny, efficientnet_b3, resnet50,
    )
    if name == 'convnext_tiny':
        m = convnext_tiny(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 4)
        return m
    if name == 'efficientnet_b3':
        m = efficientnet_b3(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 4)
        return m
    if name == 'resnet50':
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 4)
        return m
    raise ValueError(f'Unknown v2 model: {name}')


def _load_v2_model(name: str) -> _V2Wrapper:
    ckpt_path = Path(__file__).parent / 'models' / 'v2' / f'{name}.pth'
    base = _build_v2_torchvision(name)
    base.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    base.eval()
    return _V2Wrapper(base)


def get_models():
    global _models, _device
    if _models:
        return _models, _device
    _device = torch.device('cpu')

    v2_dir = Path(__file__).parent / 'models' / 'v2'
    v2_files = ['convnext_tiny.pth', 'efficientnet_b3.pth', 'resnet50.pth']
    if v2_dir.exists() and all((v2_dir / f).exists() for f in v2_files):
        # ── Production path: v2 weights (ConvNeXt-Tiny + EfficientNet-B3 + ResNet-50)
        print('[get_models] loading v2 weights from models/v2/', flush=True)
        for key in ('convnext_tiny', 'efficientnet_b3', 'resnet50'):
            print(f"  loading {key}...", flush=True)
            _models[key] = _load_v2_model(key).to(_device)
        print(f'[get_models] v2 ensemble loaded ({len(_models)} models)', flush=True)
        return _models, _device

    # ── Legacy fallback: v1 keys / v1 weights
    print('[get_models] v2 weights missing - falling back to v1 weights', flush=True)
    print("Loading DenseNet-169...", flush=True)
    _models['densenet169'] = _load_model(DenseNet169Classifier, 'densenet169_best.pth').to(_device)
    print("Loading EfficientNet-B3...", flush=True)
    _models['efficientnetb3'] = _load_model(EfficientNetB3Classifier, 'efficientnetb3_best.pth').to(_device)
    resnet_ckpt = Path(__file__).parent / 'models' / 'resnet50_best.pth'
    if resnet_ckpt.exists():
        print("Loading ResNet-50...", flush=True)
        _models['resnet50'] = _load_model(ResNet50Classifier, 'resnet50_best.pth').to(_device)
        print("All 3 models loaded!", flush=True)
    else:
        print("ResNet-50 checkpoint not found - running 2-model ensemble.", flush=True)
    return _models, _device


# Lazy-loaded OOD calibration (energy-score thresholds from val set).
_tumor_ood_calibration = None


def get_tumor_ood_calibration():
    """Load energy-score OOD thresholds for the tumor models (per-model)."""
    global _tumor_ood_calibration
    if _tumor_ood_calibration is not None:
        return _tumor_ood_calibration
    import json as _json
    path = Path(__file__).parent / 'models' / 'tumor_ood_calibration.json'
    try:
        _tumor_ood_calibration = _json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"WARN: could not load tumor OOD calibration ({e}); falling back to disabled OOD.", flush=True)
        _tumor_ood_calibration = {}
    return _tumor_ood_calibration


# ── Helpers ─────────────────────────────────────────────────────────
def get_target_layer(model, model_name):
    # v1 / v2 DenseNet — kept for the legacy fallback path
    if model_name == 'densenet169':
        return model.base_model.features.denseblock4.denselayer32.conv2
    # v2 ConvNeXt-Tiny: last CNBlock stage (features[7])
    if model_name == 'convnext_tiny':
        return model.base_model.features[7]
    # v1 + v2 ResNet-50
    if model_name == 'resnet50':
        return model.base_model.layer4[-1].conv3
    # v1 + v2 EfficientNet-B3 (last stage is features[-1] in both)
    return model.base_model.features[-1]


def get_layercam_layers(model, model_name):
    if model_name == 'densenet169':
        return {
            'block1': model.base_model.features.denseblock1,
            'block2': model.base_model.features.denseblock2,
            'block3': model.base_model.features.denseblock3,
            'block4': model.base_model.features.denseblock4,
        }
    if model_name == 'convnext_tiny':
        # torchvision ConvNeXt-Tiny features layout:
        # 0=stem, 1=stage1 (CNBlock x3), 2=ds, 3=stage2 (x3), 4=ds, 5=stage3 (x9), 6=ds, 7=stage4 (x3)
        return {
            'block1': model.base_model.features[1],
            'block2': model.base_model.features[3],
            'block3': model.base_model.features[5],
            'block4': model.base_model.features[7],
        }
    if model_name == 'resnet50':
        return {
            'block1': model.base_model.layer1,
            'block2': model.base_model.layer2,
            'block3': model.base_model.layer3,
            'block4': model.base_model.layer4,
        }
    # EfficientNet-B3 (v1 + v2)
    return {
        'block1': model.base_model.features[2],
        'block2': model.base_model.features[4],
        'block3': model.base_model.features[6],
        'block4': model.base_model.features[8],
    }


def _validate_and_enhance(image):
    info = {'original_size': list(image.size), 'enhanced': False, 'warnings': []}
    w, h = image.size
    if w < 32 or h < 32:
        info['warnings'].append(f'Very small image ({w}x{h}), results may be unreliable')
    arr = np.array(image)
    gray = np.mean(arr, axis=2) if arr.ndim == 3 else arr.astype(float)
    std_val = float(gray.std())
    mean_val = float(gray.mean())
    info['contrast'] = round(std_val, 2)
    info['brightness'] = round(mean_val, 2)
    if std_val < 10:
        info['warnings'].append('Very low contrast image')
    # CLAHE is ALWAYS applied at inference. The v2 ensemble was trained with
    # Albumentations CLAHE at p=0.5 (clipLimit=2.0, tileGrid=8x8 — identical
    # parameters), so the models see both raw and CLAHE-enhanced samples
    # during training and handle both at inference. Always-on at inference
    # gives consistent behaviour without inflating low-contrast scans
    # differently from well-lit ones.
    arr_uint8 = arr if arr.dtype == np.uint8 else (arr * 255).astype(np.uint8)
    if arr_uint8.ndim == 3:
        lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(arr_uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(enhanced)
    info['enhanced'] = True
    return image, info


def _preprocess_pil(pil_img, cross_scanner: bool = False):
    """Apply the inference preprocessing pipeline to a PIL image.

    cross_scanner=False  (Standard mode):
        RGB → adaptive CLAHE → Resize → ToTensor → ImageNet normalize.
        Matches the training distribution of the combined dataset.
        Best accuracy on BRISC / Kaggle / Epic-CSCR scanner images.

    cross_scanner=True  (Cross-Scanner mode):
        RGB → brain extraction+crop → sequence detection → Nyúl histogram
        matching → adaptive CLAHE → Resize → ToTensor → ImageNet normalize.
        Better for images from different clinics / scanner models not in the
        training set.  Phase B retrain will close the remaining train/inference
        gap — until then there is a slight accuracy cost on in-distribution images.
    """
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    sequence_warning = None
    cross_scanner_applied = False

    if cross_scanner:
        try:
            from preprocessing.brain_preprocess import (
                brain_extract_and_crop, nyul_normalize, detect_mri_sequence
            )
            # Detect sequence type BEFORE Nyúl (needs original intensities)
            seq_result = detect_mri_sequence(pil_img)
            if seq_result.get('is_non_t1'):
                sequence_warning = {
                    'is_non_t1':  True,
                    'cv':         seq_result.get('cv'),
                    'mean_brain': seq_result.get('mean_brain'),
                    'message':    seq_result.get('warning_msg'),
                }
            pil_img = brain_extract_and_crop(pil_img, padding=0.08)
            pil_img = nyul_normalize(pil_img)
            cross_scanner_applied = True
        except Exception as e:
            print(f'[preprocess] cross-scanner preprocessing unavailable: {e}', flush=True)

    pil_img, prep_info = _validate_and_enhance(pil_img)
    prep_info['cross_scanner_preprocessing'] = cross_scanner_applied
    prep_info['sequence_warning'] = sequence_warning
    prep_info['mode'] = 'cross_scanner' if cross_scanner else 'standard'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0), pil_img, prep_info


def preprocess_image(image_bytes, cross_scanner: bool = False):
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError('Cannot open file as image. Please upload a valid JPEG/PNG.')
    return _preprocess_pil(image, cross_scanner=cross_scanner)


def crop_to_tumor(original_img, bbox_224, padding_frac=0.05, min_size=32):
    """Crop the original PIL image to the bbox produced in 224x224 frame, with padding."""
    if bbox_224 is None:
        return None, None
    W, H = original_img.size
    sx = W / 224.0
    sy = H / 224.0
    x = float(bbox_224['x']) * sx
    y = float(bbox_224['y']) * sy
    w = float(bbox_224['w']) * sx
    h = float(bbox_224['h']) * sy
    pad_w = w * padding_frac
    pad_h = h * padding_frac
    left   = max(0,    int(round(x - pad_w)))
    top    = max(0,    int(round(y - pad_h)))
    right  = min(W,    int(round(x + w + pad_w)))
    bottom = min(H,    int(round(y + h + pad_h)))
    if right - left < min_size or bottom - top < min_size:
        return None, None
    crop = original_img.crop((left, top, right, bottom))
    bbox_orig = {'x': left, 'y': top, 'w': right - left, 'h': bottom - top}
    return crop, bbox_orig


def _generate_anatomy_views(original_img, target_size=256):
    """Return a dict of base64-PNG anatomy views in different color schemes.

    Helps the clinician see tissue contrasts that a single grayscale view hides
    (enhancement → HOT, bone-like edges → BONE, subtle variations → VIRIDIS,
    inverted lesion → INV). Used only in the Advanced MedGemma report.

    Returns: { 'original': b64, 'inverted': b64, 'hot': b64, 'jet': b64,
               'bone': b64, 'viridis': b64 }  — each at ~256px wide.
    """
    try:
        # Resize once, keep aspect ratio
        pil = original_img.convert('RGB')
        w, h = pil.size
        if max(w, h) > target_size:
            scale = target_size / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        rgb = np.array(pil)                          # H×W×3 uint8
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) # H×W

        def _encode(np_img_bgr_or_rgb, is_rgb=True):
            arr = np_img_bgr_or_rgb if not is_rgb else cv2.cvtColor(np_img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
            ok, png = cv2.imencode('.png', arr)
            if not ok:
                return None
            return base64.b64encode(png.tobytes()).decode('utf-8')

        def _colormap(cv_map):
            return cv2.applyColorMap(gray, cv_map)   # returns BGR

        views = {
            'original': _encode(rgb, is_rgb=True),
            'inverted': _encode(255 - rgb, is_rgb=True),
            'hot':      _encode(_colormap(cv2.COLORMAP_HOT),     is_rgb=False),
            'jet':      _encode(_colormap(cv2.COLORMAP_JET),     is_rgb=False),
            'bone':     _encode(_colormap(cv2.COLORMAP_BONE),    is_rgb=False),
            'viridis':  _encode(_colormap(cv2.COLORMAP_VIRIDIS), is_rgb=False),
        }
        return views
    except Exception as e:
        print(f'[anatomy] generation failed: {e}', flush=True)
        return None


def display_crop_with_bbox(original_img, bbox_224, padding_frac=1.0, min_side=200):
    """Build the *display* crop shown in the UI: wider than the focus-classifier
    crop (so the tumor has visible context) AND with a bounding box drawn on it
    showing where the tumor sits inside the crop.

    Returns (PIL.Image, crop_origin_dict, tumor_box_in_crop_dict) or (None, None, None).
    """
    if bbox_224 is None:
        return None, None, None
    W, H = original_img.size
    sx = W / 224.0
    sy = H / 224.0
    tx = int(round(float(bbox_224['x']) * sx))
    ty = int(round(float(bbox_224['y']) * sy))
    tw = int(round(float(bbox_224['w']) * sx))
    th = int(round(float(bbox_224['h']) * sy))

    # Padding-fraction-of-tumor expansion + minimum side guarantee so small
    # lesions don't render at 60x80 px in the UI.
    pad_w = tw * padding_frac
    pad_h = th * padding_frac
    need_w = max(0, min_side - (tw + 2 * pad_w)) / 2.0
    need_h = max(0, min_side - (th + 2 * pad_h)) / 2.0
    pad_w += need_w
    pad_h += need_h

    left   = max(0, int(round(tx - pad_w)))
    top    = max(0, int(round(ty - pad_h)))
    right  = min(W, int(round(tx + tw + pad_w)))
    bottom = min(H, int(round(ty + th + pad_h)))
    if right - left < 32 or bottom - top < 32:
        return None, None, None

    crop = original_img.crop((left, top, right, bottom)).convert('RGB').copy()

    # Tumor bbox translated into crop-local coordinates
    bx = tx - left
    by = ty - top
    bw, bh = tw, th
    arr = np.array(crop)
    cv2.rectangle(arr, (bx, by), (bx + bw, by + bh), (50, 220, 50), 2)
    # Subtle corner ticks for clarity at small sizes
    tick = max(3, min(8, min(bw, bh) // 8))
    for (cx, cy) in [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]:
        cv2.line(arr, (cx - tick, cy), (cx + tick, cy), (50, 220, 50), 2)
        cv2.line(arr, (cx, cy - tick), (cx, cy + tick), (50, 220, 50), 2)
    crop = Image.fromarray(arr)

    return crop, {'x': left, 'y': top, 'w': right - left, 'h': bottom - top}, \
           {'x': bx, 'y': by, 'w': bw, 'h': bh}


def numpy_to_base64(arr):
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def cam_to_heatmap_b64(cam, original_img=None, alpha=0.4):
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if original_img is not None:
        orig = np.array(original_img.resize((224, 224))).astype(np.float32)
        overlay = ((1 - alpha) * orig + alpha * heatmap.astype(np.float32))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return numpy_to_base64(overlay)
    return numpy_to_base64(heatmap)


# ── Novel #1: Uncertainty-Aware XAI (MC Dropout, T=20) ──────────────
def predict_with_uncertainty(model, image_tensor, device, T=20, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES
    n_classes = len(class_names)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    image_tensor = image_tensor.to(device)
    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(image_tensor)
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy()[0])
    model.eval()
    all_probs = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    var_probs = all_probs.var(axis=0)
    pred_class = int(mean_probs.argmax())
    pred_prob = float(mean_probs[pred_class])
    epistemic = float(np.sqrt(var_probs[pred_class]))
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    aleatoric = float(entropy / np.log(n_classes))
    total = float(np.sqrt(epistemic ** 2 + aleatoric ** 2))
    z = scipy_stats.norm.ppf(0.975)
    std = float(np.std(all_probs[:, pred_class]))
    ci_lower = max(0.0, pred_prob - z * std)
    ci_upper = min(1.0, pred_prob + z * std)
    needs_review = epistemic > 0.10 or pred_prob < 0.7 or (ci_upper - ci_lower) > 0.4
    return {
        'prediction': pred_class,
        'class_name': class_names[pred_class],
        'mean_confidence': pred_prob,
        'epistemic': round(epistemic, 6),
        'aleatoric': round(aleatoric, 6),
        'total_uncertainty': round(total, 6),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'needs_review': bool(needs_review),
        'probabilities': {class_names[i]: round(float(mean_probs[i]), 6) for i in range(n_classes)},
    }


# ── Novel #3: Hierarchical 4-Level XAI ─────────────────────────────
def _gradcam(model, image_tensor, target_layer, target_class):
    activations, gradients = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: activations.update({'v': o.detach()}))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.update({'v': go[0].detach()}))
    try:
        model.eval()
        inp = image_tensor.clone().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1
        out.backward(gradient=one_hot)
        w = gradients['v'].mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * activations['v']).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    finally:
        h1.remove()
        h2.remove()


def _gradcam_pp(model, image_tensor, target_layer, target_class):
    activations, gradients = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: activations.update({'v': o.detach()}))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.update({'v': go[0].detach()}))
    try:
        model.eval()
        inp = image_tensor.clone().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1
        out.backward(gradient=one_hot)
        grads = gradients['v'][0]
        acts = activations['v'][0]
        g2 = grads ** 2
        g3 = grads ** 3
        s = torch.sum(acts, dim=(1, 2), keepdim=True)
        alpha = g2 / (2 * g2 + s * g3 + 1e-10)
        weights = torch.sum(alpha * F.relu(grads), dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], device=acts.device)
        for ww, aa in zip(weights, acts):
            cam += ww * aa
        cam = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
    finally:
        h1.remove()
        h2.remove()


def _layercam(model, image_tensor, layers_dict, target_class):
    activations, gradients = {}, {}
    hooks = []
    for name, layer in layers_dict.items():
        def fwd(n):
            return lambda m, i, o: activations.update({n: o.detach()})
        def bwd(n):
            return lambda m, gi, go: gradients.update({n: go[0].detach()})
        hooks.append(layer.register_forward_hook(fwd(name)))
        hooks.append(layer.register_full_backward_hook(bwd(name)))
    try:
        model.eval()
        inp = image_tensor.clone().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1
        out.backward(gradient=one_hot)
        layer_cams = {}
        for name in layers_dict:
            if name not in gradients:
                continue
            g = gradients[name][0]
            a = activations[name][0]
            cam = F.relu(g * a).sum(dim=0).cpu().numpy()
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            cam = cv2.resize(cam, (224, 224))
            layer_cams[name] = cam
        if layer_cams:
            cams = list(layer_cams.values())
            w = np.linspace(0.3, 1.0, len(cams))
            w /= w.sum()
            fused = np.average(cams, axis=0, weights=w)
        else:
            fused = np.zeros((224, 224))
        return layer_cams, fused
    finally:
        for h in hooks:
            h.remove()


def _tumor_questions(target_class):
    is_tumor = CLASS_NAMES[target_class] != 'No Tumor'
    return {
        'level1': 'Is there a tumor?'        if is_tumor else 'Why is there no tumor?',
        'level2': 'What type of tumor?'      if is_tumor else 'What makes this brain normal?',
        'level3': 'Where is the tumor?'      if is_tumor else 'Which regions confirm normalcy?',
        'level4': 'How does the model reason?' if is_tumor else 'How does the model confirm no tumor?',
    }


def hierarchical_xai(model, model_name, image_tensor, device, target_class,
                     original_img=None, questions=None):
    """Run 4-level XAI (Grad-CAM, Grad-CAM++, LayerCAM block4, fused LayerCAM).

    `questions` is a dict with keys 'level1'..'level4' for task-specific text.
    Defaults to the tumor-pipeline questions for backward compatibility.
    """
    image_tensor = image_tensor.to(device)
    tl = get_target_layer(model, model_name)
    layers = get_layercam_layers(model, model_name)
    cam_l1 = _gradcam(model, image_tensor, tl, target_class)
    cam_l2 = _gradcam_pp(model, image_tensor, tl, target_class)
    layer_cams, fused = _layercam(model, image_tensor, layers, target_class)
    if questions is None:
        questions = _tumor_questions(target_class)
    xai_result = {
        'levels': {
            'level1_detection': {
                'name': 'Detection (Grad-CAM)',
                'question': questions['level1'],
                'heatmap': cam_to_heatmap_b64(cam_l1, original_img)
            },
            'level2_classification': {
                'name': 'Classification (Grad-CAM++)',
                'question': questions['level2'],
                'heatmap': cam_to_heatmap_b64(cam_l2, original_img)
            },
            'level3_localization': {
                'name': 'Localization (LayerCAM Block4)',
                'question': questions['level3'],
                'heatmap': cam_to_heatmap_b64(layer_cams.get('block4', np.zeros((224, 224))), original_img)
            },
            'level4_deep': {
                'name': 'Deep Analysis (LayerCAM Fused)',
                'question': questions['level4'],
                'heatmap': cam_to_heatmap_b64(fused, original_img)
            }
        }
    }
    raw_cams = {'gradcam': cam_l1, 'gradcam_pp': cam_l2, 'layercam_fused': fused}
    return xai_result, raw_cams


# ── Malignancy Scoring (type-based risk × tumor size × confidence) ─
# Per-class clinical context shown alongside the malignancy score. Static
# medical knowledge from standard neuroradiology references — used to give the
# malignancy card meaningful "what does this score mean for THIS tumor type"
# context without depending on the LLM. EN + ES variants keyed below.

CLINICAL_CONTEXT_ES = {
    'Glioma': {
        'primer': ('Los gliomas surgen de células gliales (astrocitos, oligodendrocitos). '
                   'A menudo son infiltrativos — los bordes pueden extenderse más allá de lo visible. '
                   'Los grados van de I a IV; III–IV (anaplásico / glioblastoma) son altamente malignos.'),
        'size_interpretation': {
            'small':  'Foco pequeño con realce — podría ser etapa temprana o bajo grado. Igualmente requiere seguimiento cercano.',
            'medium': 'Volumen moderado — común en gliomas de grado II–III.',
            'large':  'Masa grande — preocupante por alto grado (glioblastoma) con efecto masa; revisión neuroquirúrgica urgente.',
        },
        'subtypes': ['Astrocitoma', 'Oligodendroglioma', 'Glioblastoma (GBM)'],
        'typical_workup': 'RM con contraste, espectroscopia por RM, biopsia estereotáctica o resección quirúrgica para histopatología.',
    },
    'Meningioma': {
        'primer': ('Los meningiomas surgen de las células aracnoideas de las meninges. '
                   'Mayormente benignos (OMS grado I, ~90%). De crecimiento lento. Usualmente extra-axiales '
                   'con amplia inserción dural.'),
        'size_interpretation': {
            'small':  'Lesión pequeña — los meningiomas incidentales <2 cm a menudo se vigilan, no se operan.',
            'medium': 'Tamaño moderado — sintomático según la ubicación; se considera cirugía con frecuencia.',
            'large':  'Masa grande — probablemente sintomática (cefalea, déficits focales) pero el grado aún puede ser benigno.',
        },
        'subtypes': ['OMS grado I (benigno)', 'OMS grado II (atípico)', 'OMS grado III (anaplásico, raro)'],
        'typical_workup': 'RM con contraste (buscar cola dural), conducta expectante si es pequeño y asintomático, resección quirúrgica si crece o causa síntomas.',
    },
    'Pituitary': {
        'primer': ('Los adenomas hipofisarios surgen en la silla turca. Casi siempre benignos. '
                   'Clasificados por tamaño: microadenoma <10 mm, macroadenoma ≥10 mm. '
                   'Pueden ser funcionantes (secretores de hormonas) o no funcionantes.'),
        'size_interpretation': {
            'small':  'Probablemente microadenoma (<10 mm). A menudo secretor hormonal (prolactina, ACTH, GH).',
            'medium': 'Macroadenoma — puede causar déficits del campo visual si comprime el quiasma óptico.',
            'large':  'Macroadenoma grande — riesgo de compresión quiasmática, hipopituitarismo; derivación a neurocirugía + endocrinología.',
        },
        'subtypes': ['Adenoma no funcionante', 'Prolactinoma', 'Secretor de GH (acromegalia)', 'Secretor de ACTH (Cushing)'],
        'typical_workup': 'RM con protocolo hipofisario (cortes finos sellares), panel hormonal (prolactina, IGF-1, cortisol, TSH), campimetría visual formal si es macroadenoma.',
    },
}


CLINICAL_CONTEXT = {
    'Glioma': {
        'primer': ('Gliomas arise from glial cells (astrocytes, oligodendrocytes). '
                   'Often infiltrative — borders may extend beyond what is visible. '
                   'Grades range I–IV; III–IV (anaplastic / glioblastoma) are highly malignant.'),
        'size_interpretation': {
            'small':  'Small enhancing focus — could be early-stage or low-grade. Still warrants close follow-up.',
            'medium': 'Moderate volume — common in grade II–III gliomas.',
            'large':  'Large mass — concerning for high-grade (glioblastoma) with mass effect; urgent neurosurgical review.',
        },
        'subtypes': ['Astrocytoma', 'Oligodendroglioma', 'Glioblastoma (GBM)'],
        'typical_workup': 'Contrast-enhanced MRI, MR spectroscopy, stereotactic biopsy or surgical resection for histopathology.',
    },
    'Meningioma': {
        'primer': ('Meningiomas arise from arachnoid cap cells of the meninges. '
                   'Mostly benign (WHO grade I, ~90%). Slow-growing. Usually extra-axial '
                   'with broad dural attachment.'),
        'size_interpretation': {
            'small':  'Small lesion — incidental meningiomas <2 cm are often watched, not operated.',
            'medium': 'Moderate size — symptomatic depending on location; surgery often considered.',
            'large':  'Large mass — likely symptomatic (headache, focal deficits) but grade can still be benign.',
        },
        'subtypes': ['WHO grade I (benign)', 'WHO grade II (atypical)', 'WHO grade III (anaplastic, rare)'],
        'typical_workup': 'MRI with contrast (look for dural tail), watchful waiting if small + asymptomatic, surgical resection if growing or symptomatic.',
    },
    'Pituitary': {
        'primer': ('Pituitary adenomas arise in the sella turcica. Almost always benign. '
                   'Classified by size: microadenoma <10 mm, macroadenoma ≥10 mm. '
                   'May be functioning (hormone-secreting) or non-functioning.'),
        'size_interpretation': {
            'small':  'Likely microadenoma (<10 mm). Often hormone-secreting (prolactin, ACTH, GH).',
            'medium': 'Macroadenoma — may cause visual field deficits if compressing the optic chiasm.',
            'large':  'Large macroadenoma — risk of chiasmal compression, hypopituitarism; neurosurgery + endocrinology referral.',
        },
        'subtypes': ['Non-functioning adenoma', 'Prolactinoma', 'GH-secreting (acromegaly)', 'ACTH-secreting (Cushing)'],
        'typical_workup': 'Pituitary-protocol MRI with thin sella cuts, hormone panel (prolactin, IGF-1, cortisol, TSH), formal visual field testing if macroadenoma.',
    },
}


BASE_RISK = {
    'Glioma':     {'score': 7.5, 'label': 'HIGH',       'note': 'Most gliomas are infiltrative; grades III-IV are highly malignant.'},
    'Meningioma': {'score': 3.5, 'label': 'LOW-MEDIUM', 'note': 'Majority are WHO grade I (benign); surgical resection usually curative.'},
    'Pituitary':  {'score': 2.0, 'label': 'LOW',        'note': 'Almost always benign adenomas; managed medically or with surgery.'},
    'No Tumor':   {'score': 0.0, 'label': 'N/A',        'note': 'No tumor detected.'},
}

BASE_RISK_ES = {
    'Glioma':     {'score': 7.5, 'label': 'ALTO',       'note': 'La mayoría de los gliomas son infiltrativos; los grados III-IV son altamente malignos.'},
    'Meningioma': {'score': 3.5, 'label': 'BAJO-MEDIO', 'note': 'La mayoría son OMS grado I (benignos); la resección quirúrgica suele ser curativa.'},
    'Pituitary':  {'score': 2.0, 'label': 'BAJO',       'note': 'Casi siempre adenomas benignos; manejo médico o quirúrgico.'},
    'No Tumor':   {'score': 0.0, 'label': 'N/A',        'note': 'No se detectó tumor.'},
}


def pick_lang(lang):
    """Normalize lang to 'en'|'es' (default en)."""
    return 'es' if (lang or '').lower().startswith('es') else 'en'


def compute_urgency(class_name: str, score: float, size_pct: float | None, lang='en'):
    """Derive a urgent / soon / routine / reassuring badge from class + score + size.

    This is a deterministic clinical-prior-driven function; it does not depend
    on MedGemma. Used to render the urgency banner at the top of the Diagnostic
    Impression card and embedded in the structured medical_codes response.
    """
    is_es = pick_lang(lang) == 'es'
    LABELS = {
        'urgent':   ('Urgente',  'Urgent')[0 if is_es else 1],
        'soon':     ('Pronto',   'Soon')[0 if is_es else 1],
        'routine':  ('Rutina',   'Routine')[0 if is_es else 1],
        'reassure': ('Tranquilo','Reassuring')[0 if is_es else 1],
    }
    ACTIONS_ES = {
        'urgent':   'Solicitar RM con contraste y derivación a neurocirugía hoy o mañana.',
        'soon':     'Pedir cita con neurólogo en las próximas 1-2 semanas.',
        'routine':  'Compartir las imágenes con el médico de cabecera para seguimiento programado.',
        'reassure': 'No se requiere acción urgente. Revisión de control si hay síntomas.',
    }
    ACTIONS_EN = {
        'urgent':   'Request contrast MRI and neurosurgical referral within 24-48h.',
        'soon':     'Book a neurologist appointment within 1-2 weeks.',
        'routine':  'Share these images with your primary care doctor for scheduled follow-up.',
        'reassure': 'No urgent action needed. Repeat scan only if symptoms appear.',
    }
    actions = ACTIONS_ES if is_es else ACTIONS_EN

    # Decision logic
    if class_name == 'No Tumor':
        level = 'reassure'
    elif class_name == 'Glioma':
        if score is not None and score >= 7.0:
            level = 'urgent'
        elif score is not None and score >= 5.0:
            level = 'soon'
        else:
            level = 'soon'    # All gliomas warrant prompt review
    elif class_name == 'Meningioma':
        # Large meningiomas with mass effect → soon; small incidental → routine
        if size_pct is not None and size_pct >= 8.0:
            level = 'soon'
        else:
            level = 'routine'
    elif class_name == 'Pituitary':
        # ≥10mm (macroadenoma) = clinical attention; smaller usually routine
        if size_pct is not None and size_pct >= 8.0:
            level = 'soon'
        else:
            level = 'routine'
    else:
        level = 'routine'

    return {
        'level':  level,
        'label':  LABELS[level],
        'action': actions[level],
        'color':  {'urgent': '#f5576c', 'soon': '#f5a623',
                   'routine': '#a3e635', 'reassure': '#38ef7d'}[level],
    }


# Medical coding suggestions per predicted class — never invent codes outside
# this table. These are widely used codes for the broad category; specific
# subtypes (e.g. C71.0 frontal lobe glioma) need a radiologist's input.
MEDICAL_CODES = {
    'Glioma': {
        'icd10': [
            {'code': 'C71.9', 'desc_en': 'Malignant neoplasm of brain, unspecified',
                              'desc_es': 'Neoplasia maligna del encéfalo, no especificada'},
            {'code': 'D43.2', 'desc_en': 'Neoplasm of uncertain behavior of brain, unspecified',
                              'desc_es': 'Neoplasia de comportamiento incierto del encéfalo'},
        ],
        'snomed': [
            {'code': '393563007', 'desc_en': 'Glioma (morphologic abnormality)',
                                   'desc_es': 'Glioma (anomalía morfológica)'},
            {'code': '254938000', 'desc_en': 'Glioblastoma multiforme of brain',
                                   'desc_es': 'Glioblastoma multiforme cerebral'},
        ],
    },
    'Meningioma': {
        'icd10': [
            {'code': 'D32.0', 'desc_en': 'Benign neoplasm of cerebral meninges',
                              'desc_es': 'Neoplasia benigna de meninges cerebrales'},
            {'code': 'D42.0', 'desc_en': 'Neoplasm of uncertain behavior of cerebral meninges',
                              'desc_es': 'Neoplasia de comportamiento incierto de meninges'},
        ],
        'snomed': [
            {'code': '254897005', 'desc_en': 'Meningioma',
                                   'desc_es': 'Meningioma'},
            {'code': '443329004', 'desc_en': 'Atypical meningioma',
                                   'desc_es': 'Meningioma atípico'},
        ],
    },
    'Pituitary': {
        'icd10': [
            {'code': 'D35.2', 'desc_en': 'Benign neoplasm of pituitary gland',
                              'desc_es': 'Neoplasia benigna de la glándula hipofisaria'},
            {'code': 'E22.0', 'desc_en': 'Acromegaly and pituitary gigantism (if GH-secreting)',
                              'desc_es': 'Acromegalia (si secreta GH)'},
        ],
        'snomed': [
            {'code': '254956000', 'desc_en': 'Adenoma of pituitary',
                                   'desc_es': 'Adenoma de hipófisis'},
            {'code': '237679004', 'desc_en': 'Pituitary microadenoma',
                                   'desc_es': 'Microadenoma hipofisario'},
        ],
    },
    'No Tumor': {'icd10': [], 'snomed': []},
}


def get_medical_codes(class_name, lang='en'):
    is_es = pick_lang(lang) == 'es'
    codes = MEDICAL_CODES.get(class_name, MEDICAL_CODES['No Tumor'])
    out = {'icd10': [], 'snomed': []}
    for c in codes['icd10']:
        out['icd10'].append({'code': c['code'], 'desc': c['desc_es' if is_es else 'desc_en']})
    for c in codes['snomed']:
        out['snomed'].append({'code': c['code'], 'desc': c['desc_es' if is_es else 'desc_en']})
    return out


def get_base_risk(class_name, lang='en'):
    table = BASE_RISK_ES if pick_lang(lang) == 'es' else BASE_RISK
    return table.get(class_name, table['No Tumor'])


# 5%-wide buckets for the displayed tumor-size estimate. Exact percentages from
# the pixel pipeline are noisy (segmentation choices swing the number ~2-5%
# between runs), so the UI/report shows a range like "10-15%" instead of a
# spurious-precision "12.3%". The raw pct is kept in the response for anyone
# who wants the underlying number.
def size_pct_to_range(size_pct):
    """size_pct in the 0-100 range. Returns (low, high, label) or None."""
    if size_pct is None:
        return None
    try:
        s = float(size_pct)
    except (TypeError, ValueError):
        return None
    if s <= 0:
        return None
    if s < 1.0:
        return (0.0, 1.0, '<1%')
    if s < 5.0:
        return (1.0, 5.0, '1-5%')
    if s >= 30.0:
        return (30.0, 100.0, '30%+')
    low = int(s // 5) * 5
    high = low + 5
    return (float(low), float(high), f'{low}-{high}%')


def get_clinical_context(class_name, lang='en'):
    table = CLINICAL_CONTEXT_ES if pick_lang(lang) == 'es' else CLINICAL_CONTEXT
    return table.get(class_name)


# ── SAM-based tumor segmentation (replaces hybrid CAM x intensity sizing) ─
# MobileSAM is small (~38 MB) and produces pixel-tight masks from a bbox prompt.
# We sentinel-cache the loader: None = not tried, False = tried and failed,
# anything else = a ready model instance.
_sam_model = None


def _get_sam():
    global _sam_model
    if _sam_model is None:
        try:
            from ultralytics import SAM
            ckpt = Path(__file__).parent / 'models' / 'mobile_sam.pt'
            print('[SAM] loading MobileSAM (first call auto-downloads ~38 MB) ...', flush=True)
            _sam_model = SAM(str(ckpt) if ckpt.exists() else 'mobile_sam.pt')
            print('[SAM] ready', flush=True)
        except Exception as e:
            print(f'[SAM] disabled (falling back to hybrid CAM sizing): {e}', flush=True)
            _sam_model = False
    return _sam_model if _sam_model is not False else None


# ── YOLO tumor detector (trained on Cheng et al. 2015, 3,064 slices) ─────
# Single-class detector that returns a tumor bbox directly. Replaces the CAM
# heuristics as the primary localization signal — validated mAP50=0.918 on
# a patient-disjoint held-out set of 47 patients.
# Sentinel-cached the same way as SAM: None=not tried, False=failed, else=model.
_yolo_model = None
YOLO_WEIGHTS = Path(__file__).parent / 'cheng_yolo.pt'
YOLO_MIN_CONF = 0.30  # below this we don't trust the detector


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            if not YOLO_WEIGHTS.exists():
                raise FileNotFoundError(f'{YOLO_WEIGHTS} not found')
            print(f'[YOLO] loading {YOLO_WEIGHTS.name} ...', flush=True)
            _yolo_model = YOLO(str(YOLO_WEIGHTS))
            print('[YOLO] ready', flush=True)
        except Exception as e:
            print(f'[YOLO] disabled (falling back to CAM): {e}', flush=True)
            _yolo_model = False
    return _yolo_model if _yolo_model is not False else None


def _yolo_detect_bbox_224(pil_image):
    """Run the trained YOLO detector on the original image, return the best bbox
    in 224x224 coordinates plus its confidence, or None if no detection.

    The model was trained at imgsz=640 on slices that are typically 512x512
    grayscale T1c. We pass the original PIL image directly and let Ultralytics
    handle resizing/padding internally, then remap the predicted xyxy back to
    our 224x224 working frame.
    """
    yolo = _get_yolo()
    if yolo is None:
        return None
    try:
        img = pil_image.convert('RGB')
        W, H = img.size
        res = yolo.predict(np.array(img), imgsz=640, conf=YOLO_MIN_CONF,
                           verbose=False, device='cpu')[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        # Pick the box with highest confidence
        confs = res.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        x0, y0, x1, y1 = res.boxes.xyxy[idx].cpu().numpy().astype(float).tolist()
        conf = float(confs[idx])
        # Remap to 224x224
        sx, sy = 224.0 / W, 224.0 / H
        x0_r, y0_r = int(round(x0 * sx)), int(round(y0 * sy))
        x1_r, y1_r = int(round(x1 * sx)), int(round(y1 * sy))
        x0_r = max(0, min(223, x0_r)); y0_r = max(0, min(223, y0_r))
        x1_r = max(x0_r + 1, min(224, x1_r)); y1_r = max(y0_r + 1, min(224, y1_r))
        return {
            'x': x0_r, 'y': y0_r,
            'w': x1_r - x0_r, 'h': y1_r - y0_r,
            'conf': conf,
        }
    except Exception as e:
        print(f'[YOLO] predict failed: {e}', flush=True)
        return None


def _cam_top_k_bboxes(cam, brain_mask, k=3, percentile=90, min_area=20):
    """Extract up to K bounding boxes from the most-intense Grad-CAM++ peaks.

    Threshold the CAM at the given percentile, intersect with the brain mask,
    find connected components, and rank each one by (mean activation × area^0.3).
    Returns up to K bbox dicts ordered most-intense first.

    This replaces the legacy "pick the largest connected component" heuristic
    which loses small-but-intense tumor peaks to large-but-weak background
    activations. Now intensity carries more weight than raw area.
    """
    cam_r = cv2.resize(cam, (224, 224)).astype(np.float32)
    pos = cam_r[cam_r > 0]
    if pos.size == 0:
        return []
    thresh = float(np.percentile(pos, percentile))
    mask = ((cam_r >= thresh).astype(np.uint8) * 255)
    mask = cv2.bitwise_and(mask, brain_mask)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return []
    scored = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cc_pixels = cam_r[lbl == i]
        if cc_pixels.size == 0:
            continue
        mean_act = float(cc_pixels.mean())
        # Balance intensity with size — area^0.3 keeps small intense peaks competitive
        score = mean_act * (float(area) ** 0.3)
        scored.append((score, {'x': x, 'y': y, 'w': w, 'h': h}))
    scored.sort(key=lambda s: -s[0])
    return [bbox for _s, bbox in scored[:k]]


def _real_brain_mask(gray_224):
    """Otsu + close + largest CC. Replaces the raw-Otsu brain-area count that
    inflated `size_pct` denominators when skull-stripped images had dim cortex
    falling under the Otsu threshold."""
    _, m = cv2.threshold(gray_224, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(m)
    out[lbl == largest] = 255
    return out


def _sam_segment_mask(pil_224, bbox):
    """Run SAM with a bbox prompt on the 224x224 image. Returns binary uint8 mask
    (224x224, values 0/255) or None on failure / SAM unavailable."""
    sam = _get_sam()
    if sam is None or bbox is None:
        return None
    arr = np.array(pil_224.convert('RGB'))
    x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
    try:
        res = sam(arr, bboxes=[[x, y, x + w, y + h]], verbose=False)
    except Exception as e:
        print(f'[SAM] inference failed: {e}', flush=True)
        return None
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return None
    mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    if mask.shape != (224, 224):
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    return mask


def _sam_run_safe(sam_model, arr, **kwargs):
    """Helper: run SAM with given prompts and return a clean 224x224 mask or None."""
    try:
        res = sam_model(arr, verbose=False, **kwargs)
    except Exception as e:
        print(f'[SAM] inference failed ({kwargs}): {e}', flush=True)
        return None
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return None
    mask = res[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    if mask.shape != (224, 224):
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    return mask


def _sam_segment_refined(pil_224, bbox_224, brain_mask=None, extra_bboxes=None):
    """Multi-strategy SAM segmentation with intensity-based candidate selection.

    When a single bbox prompt mis-localises (bbox slightly off the tumor → mask
    cuts off bright tumor pixels), running SAM with several complementary prompts
    and then picking the candidate that maximises intensity contrast against
    surrounding brain produces noticeably tighter and more accurate masks for
    enhancing lesions on T1.

    Strategies tried (per seed bbox):
        (a) Bbox as-is
        (b) Bbox expanded ~25 % on each side (gives SAM room to grow)
        (c) Foreground POINT at the peak-intensity pixel inside the bbox
            (tumors are usually the local bright spot → great seed)
        (d) Combined bbox + point
    Additional global fallback:
        (e) Brightest connected blob inside brain mask — independent of any bbox

    `extra_bboxes` is a list of additional bbox dicts (e.g., the Grad-CAM++
    derived bbox) — each one contributes its own (a) and (b) candidates so we
    can survive cases where the primary bbox is wrong but Grad-CAM++ is right.

    Score = (mean intensity inside − mean intensity outside) / outside σ
            + mild bonus for plausible size, penalty for excessive size

    Returns (binary uint8 mask 224x224, strategy_name) or (None, None).
    """
    sam = _get_sam()
    if sam is None or bbox_224 is None:
        return None, None

    arr = np.array(pil_224.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
    if brain_mask is None:
        brain_mask = _real_brain_mask(gray)
    brain_area = int((brain_mask > 0).sum())
    if brain_area == 0:
        return None, None

    x = int(bbox_224['x']); y = int(bbox_224['y'])
    w = int(bbox_224['w']); h = int(bbox_224['h'])
    x = max(0, min(223, x)); y = max(0, min(223, y))
    w = max(8, min(224 - x, w)); h = max(8, min(224 - y, h))

    # Expanded bbox (+25 % on each side, clipped to image bounds)
    ex_pad = max(4, int(round(w * 0.25)))
    ey_pad = max(4, int(round(h * 0.25)))
    ex1 = max(0, x - ex_pad); ey1 = max(0, y - ey_pad)
    ex2 = min(224, x + w + ex_pad); ey2 = min(224, y + h + ey_pad)

    # Find peak-intensity foreground point inside the bbox (most likely tumor core
    # on T1-enhancing lesions, which appear bright). Use Gaussian-blurred grayscale
    # to be robust to single noisy pixels.
    peak_xy = None
    roi_gray = gray[y:y + h, x:x + w]
    roi_brain = brain_mask[y:y + h, x:x + w]
    if roi_gray.size > 0 and (roi_brain > 0).any():
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0).astype(np.int32)
        blurred[roi_brain == 0] = -1
        _, _, _, max_loc = cv2.minMaxLoc(blurred.astype(np.float32))
        peak_xy = (x + int(max_loc[0]), y + int(max_loc[1]))

    candidates = []

    # (a) bbox-only
    m = _sam_run_safe(sam, arr, bboxes=[[x, y, x + w, y + h]])
    if m is not None: candidates.append(('bbox', m))

    # (b) expanded bbox
    m = _sam_run_safe(sam, arr, bboxes=[[ex1, ey1, ex2, ey2]])
    if m is not None: candidates.append(('bbox_expanded', m))

    # (c) peak-intensity point inside the bbox
    if peak_xy is not None:
        m = _sam_run_safe(sam, arr, points=[list(peak_xy)], labels=[1])
        if m is not None: candidates.append(('peak_point', m))

    # (d) combined: bbox + point
    if peak_xy is not None:
        m = _sam_run_safe(sam, arr,
                          bboxes=[[ex1, ey1, ex2, ey2]],
                          points=[list(peak_xy)],
                          labels=[1])
        if m is not None: candidates.append(('bbox_plus_point', m))

    # Extra bboxes from independent sources (e.g. Grad-CAM++'s bbox). Each one
    # gets two cheap candidates: the bbox itself and a 25 % expanded version.
    # When MedGemma's bbox is wrong but Grad-CAM++ is right (or vice versa), the
    # contrast-based scoring picks whichever produces the higher-quality mask.
    if extra_bboxes:
        for idx, eb in enumerate(extra_bboxes):
            if not eb:
                continue
            try:
                ex = max(0, min(223, int(eb['x'])))
                ey = max(0, min(223, int(eb['y'])))
                ew = max(8, min(224 - ex, int(eb['w'])))
                eh = max(8, min(224 - ey, int(eb['h'])))
            except (KeyError, TypeError, ValueError):
                continue
            tag = f'cam_bbox_{idx}'
            m = _sam_run_safe(sam, arr, bboxes=[[ex, ey, ex + ew, ey + eh]])
            if m is not None: candidates.append((tag, m))
            # Expanded version
            xp = max(4, int(round(ew * 0.25)))
            yp = max(4, int(round(eh * 0.25)))
            ex1 = max(0, ex - xp); ey1 = max(0, ey - yp)
            ex2 = min(224, ex + ew + xp); ey2 = min(224, ey + eh + yp)
            m = _sam_run_safe(sam, arr, bboxes=[[ex1, ey1, ex2, ey2]])
            if m is not None: candidates.append((tag + '_expanded', m))

    # (e) Intensity-driven fallback — INDEPENDENT of the input bbox. Finds the
    # brightest connected blob inside the brain mask (after smoothing). For T1
    # post-contrast / T1c images, enhancing tumors are typically the brightest
    # non-skull region — this strategy is the safety net when MedGemma's bbox is
    # wildly wrong (e.g., model defaults to a generic location regardless of
    # actual lesion position). If MedGemma's bbox is correct, the intensity
    # candidate will overlap and produce a similar mask; if MedGemma's bbox is
    # wrong, this candidate often saves the case via the contrast-based scoring.
    try:
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        gray_in_brain = gray_blur.copy()
        gray_in_brain[brain_mask == 0] = 0
        # Top-2 % brightest brain pixels — robust to single hot pixels
        brain_vals = gray_blur[brain_mask > 0]
        if brain_vals.size > 0:
            thresh = float(np.percentile(brain_vals, 98))
            bright = ((gray_in_brain >= thresh) & (brain_mask > 0)).astype(np.uint8) * 255
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n_b, lbl_b, stats_b, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
            if n_b > 1:
                # Pick the largest bright component (ignore tiny specks)
                i_b = 1 + int(np.argmax(stats_b[1:, cv2.CC_STAT_AREA]))
                bb_x = int(stats_b[i_b, cv2.CC_STAT_LEFT])
                bb_y = int(stats_b[i_b, cv2.CC_STAT_TOP])
                bb_w = int(stats_b[i_b, cv2.CC_STAT_WIDTH])
                bb_h = int(stats_b[i_b, cv2.CC_STAT_HEIGHT])
                bb_area = int(stats_b[i_b, cv2.CC_STAT_AREA])
                # Reject implausible (skull-strip artifacts or huge brain-half blobs)
                if 30 < bb_area < int(brain_area * 0.35):
                    cx = int(stats_b[i_b, cv2.CC_STAT_LEFT] + bb_w // 2)
                    cy = int(stats_b[i_b, cv2.CC_STAT_TOP] + bb_h // 2)
                    # (e1) bright-blob bbox
                    m = _sam_run_safe(sam, arr, bboxes=[[bb_x, bb_y, bb_x + bb_w, bb_y + bb_h]])
                    if m is not None: candidates.append(('bright_blob_bbox', m))
                    # (e2) point at bright-blob centroid
                    m = _sam_run_safe(sam, arr, points=[[cx, cy]], labels=[1])
                    if m is not None: candidates.append(('bright_blob_point', m))
    except Exception as _e:
        print(f'[SAM] bright-blob fallback failed: {_e}', flush=True)

    if not candidates:
        return None, None

    # Score every candidate. Score formula rewards intensity contrast AND
    # plausible size — tiny bright specks (orbital fat, skull marrow) get a
    # low score because their size is too small; brain-half false positives
    # get a low score because of the > 25 % size penalty.
    INTENSITY_BASED = ('bright_blob_bbox', 'bright_blob_point')
    scored = []
    for strategy, mask in candidates:
        m_clean = cv2.bitwise_and(mask, brain_mask)
        n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(m_clean, connectivity=8)
        if n_cc <= 1:
            continue
        i_cc = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
        largest = (lbl_cc == i_cc).astype(np.uint8) * 255
        area = int((largest > 0).sum())
        size_ratio = area / float(brain_area)
        if size_ratio < 0.015 or size_ratio > 0.45:
            continue
        inside = gray[largest > 0]
        outside = gray[(largest == 0) & (brain_mask > 0)]
        if inside.size == 0 or outside.size == 0:
            continue
        contrast = (float(inside.mean()) - float(outside.mean())) / (float(outside.std()) + 1e-6)
        # Reward larger plausible masks; penalise > 25 %.
        size_bonus = min(size_ratio, 0.20) * 4.0
        size_penalty = max(0.0, size_ratio - 0.25) * 8.0
        score = contrast + size_bonus - size_penalty
        scored.append((strategy, largest, score, size_ratio, contrast))

    if not scored:
        return None, None

    # Two-tier selection: prefer bbox-based candidates if any of them clears a
    # "reasonable" bar (contrast > 0 and size > 2 %). Only fall through to the
    # intensity-based bright-blob seeds when no bbox candidate is usable —
    # otherwise the bright-blob fallback can hijack a perfectly good MedGemma
    # bbox with a tiny high-contrast speck (e.g. orbital fat, calcification).
    bbox_based = [s for s in scored if s[0] not in INTENSITY_BASED]
    intensity_based = [s for s in scored if s[0] in INTENSITY_BASED]

    def pick(group):
        return max(group, key=lambda s: s[2]) if group else None

    bbox_pick = pick(bbox_based)
    intensity_pick = pick(intensity_based)

    chosen = None
    if bbox_pick is not None and bbox_pick[4] > 0.0 and bbox_pick[3] > 0.02:
        chosen = bbox_pick
    elif intensity_pick is not None:
        chosen = intensity_pick
    elif bbox_pick is not None:
        chosen = bbox_pick

    if chosen is None:
        return None, None
    best_strategy, best_mask = chosen[0], chosen[1]

    # Light morphology to clean up edges (close small gaps, then open to remove specks)
    k3 = np.ones((3, 3), np.uint8)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, k3)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, k3)
    return best_mask, best_strategy


def _extract_tumor_region(cam, original_img):
    """Tumor area estimation — SAM-prompted segmentation, hybrid CAM x intensity
    fallback if SAM is unavailable or returns an implausible mask.

    Pipeline (primary path):
      1. CAM 85th-pct + erosion -> coarse lesion bbox.
      2. MobileSAM prompted by that bbox -> pixel-accurate mask.
      3. Intersect with brain mask, take largest CC, count pixels.
      4. If size > 40 % of brain (CAM was probably pointing at noise / no real
         lesion), fall back to the hybrid method which is more conservative.

    Falls through to `_extract_tumor_region_hybrid` if any step fails. Returned
    dict adds `mask_b64` (PNG-encoded 224x224 mask) when SAM produced the answer
    so the frontend can overlay a pixel-tight boundary later.
    """
    cam_r = cv2.resize(cam, (224, 224))
    orig = np.array(original_img.resize((224, 224)))
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY) if orig.ndim == 3 else orig
    brain_mask = _real_brain_mask(gray)
    brain_area = int((brain_mask > 0).sum())
    if brain_area == 0:
        return _extract_tumor_region_hybrid(cam, original_img)

    # ── Parallel measurement at ORIGINAL resolution ─────────────────────
    # Otsu at full resolution captures the cortex/skull boundary more
    # cleanly than at 224x224 (downscaling blurs edges + biases the
    # threshold). We use this to compute a more accurate `brain_area_orig`,
    # then `size_pct_orig = tumor_area_at_orig / brain_area_orig` becomes
    # the headline estimate. Cheap — ~5 ms on a 512x512 input.
    W_orig, H_orig = original_img.size
    scale_x = W_orig / 224.0
    scale_y = H_orig / 224.0
    pixel_scale = scale_x * scale_y  # 224-pixel -> original-pixel area factor
    try:
        orig_arr_full = np.array(original_img.convert('RGB'))
        gray_full = cv2.cvtColor(orig_arr_full, cv2.COLOR_RGB2GRAY) if orig_arr_full.ndim == 3 else orig_arr_full
        brain_mask_orig = _real_brain_mask(gray_full)
        brain_area_orig = int((brain_mask_orig > 0).sum())
    except Exception as _e:
        print(f'[BRAIN] orig-resolution mask failed: {_e}', flush=True)
        brain_area_orig = 0

    # ── Primary path: trained YOLO detector ──────────────────────────
    # YOLO was supervised on radiologist-validated masks across 233 patients,
    # so its bbox is the authoritative source for the displayed box. For the
    # tumor AREA we prefer SAM's pixel-tight mask when it segments sensibly
    # (30-90% of the YOLO bbox area) — that captures the irregular outline of
    # real lesions. When SAM under- or over-segments we fall back to a
    # conservative `bbox × 0.55` estimate (calibrated to account for the
    # 10-15% padding YOLO typically leaves around each tumor).
    yolo_bbox = _yolo_detect_bbox_224(original_img)
    if yolo_bbox is not None:
        bb_area_raw   = yolo_bbox['w'] * yolo_bbox['h']
        yolo_bbox_out = {k: yolo_bbox[k] for k in ('x', 'y', 'w', 'h')}

        # Try SAM for the area measurement.
        sam_area = None
        mask_b64 = None
        pil_224 = original_img.resize((224, 224))
        try:
            sam_mask = _sam_segment_mask(pil_224, yolo_bbox_out)
            if sam_mask is not None:
                sam_mask = cv2.bitwise_and(sam_mask, brain_mask)
                n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(sam_mask, connectivity=8)
                if n_cc > 1:
                    i_cc = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
                    largest = (lbl_cc == i_cc).astype(np.uint8) * 255
                    sam_area = int((largest > 0).sum())
                    _ok, png = cv2.imencode('.png', largest)
                    if _ok:
                        mask_b64 = base64.b64encode(png.tobytes()).decode('utf-8')
        except Exception as _e:
            print(f'[SAM] mask failed: {_e}', flush=True)

        # Plausibility band for SAM. Below 0.30 → SAM under-segmented (e.g.
        # grabbed only the bright core). Above 0.90 → SAM grew past the lesion
        # into healthy tissue. In either failure mode use the calibrated bbox
        # area estimate.
        sam_ratio = (sam_area / float(bb_area_raw)) if (sam_area and bb_area_raw) else 0.0
        if sam_area is not None and 0.30 <= sam_ratio <= 0.90:
            # SAM segmented the bright enhancing core; floor-clamp to 0.80 of
            # the YOLO bbox area so we don't under-report for lesions whose
            # outer rim fades into surrounding tissue (typical meningiomas).
            # If SAM already exceeds the floor, we trust SAM's tighter measure.
            tumor_area = max(sam_area, int(bb_area_raw * 0.80))
            method = 'yolo_sam_validated'
        else:
            tumor_area = int(bb_area_raw * 0.55)
            method = 'yolo_bbox_fallback'
            # SAM mask was implausible — drop the overlay so it doesn't mislead.
            if sam_area is not None and (sam_ratio < 0.30 or sam_ratio > 0.90):
                mask_b64 = None

        # Two size_pct estimates:
        #   size_pct_224 — denominator at the model resolution (legacy)
        #   size_pct     — denominator at the original image resolution
        #                  (sharper brain mask, what we now report)
        size_pct_224 = tumor_area / float(brain_area)
        if brain_area_orig > 0:
            tumor_area_orig = int(round(tumor_area * pixel_scale))
            size_pct = tumor_area_orig / float(brain_area_orig)
        else:
            tumor_area_orig = None
            size_pct = size_pct_224

        print(f'[SIZE/YOLO] bbox_area={bb_area_raw} sam_area={sam_area} '
              f'sam_ratio={sam_ratio:.2f} tumor_area={tumor_area} '
              f'brain_area_224={brain_area} brain_area_orig={brain_area_orig} '
              f'size_pct_224={size_pct_224*100:.1f}% size_pct_orig={size_pct*100:.1f}% '
              f'method={method}', flush=True)

        if size_pct < 0.40:
            return {
                'size_pct':        size_pct,
                'size_pct_224':    round(size_pct_224, 4),
                'size_pct_orig':   round(size_pct, 4),
                'bbox':            yolo_bbox_out,
                'brain_area':      brain_area,
                'brain_area_orig': brain_area_orig,
                'tumor_area':      tumor_area,
                'tumor_area_orig': tumor_area_orig,
                'bbox_area':       bb_area_raw,
                'sam_area':        sam_area,
                'sam_ratio':       round(sam_ratio, 3) if sam_area else None,
                'method':          method,
                'yolo_conf':       yolo_bbox['conf'],
                'mask_b64':        mask_b64,
                'convention':      'radiological',
                'bbox_source':     'yolo',
            }

    pos = cam_r[cam_r > 0]
    if pos.size == 0:
        return {'size_pct': 0.0, 'bbox': None, 'brain_area': brain_area,
                'tumor_area': 0, 'method': 'no_cam', 'mask_b64': None}

    # Tight SAM seed bbox (95th pct + erode). 85th-pct from the legacy method is
    # too broad — SAM, seeded with a wide prompt, grows a mask that engulfs
    # surrounding healthy tissue. 95th-pct sits on the lesion *core*; SAM then
    # expands outward to the natural intensity boundary.
    thresh_bbox = float(np.percentile(pos, 95))
    bbox_mask = cv2.bitwise_and((cam_r >= thresh_bbox).astype(np.uint8) * 255, brain_mask)
    bbox_mask = cv2.erode(bbox_mask, np.ones((3, 3), np.uint8), iterations=1)
    n_bb, _, stats_bb, _ = cv2.connectedComponentsWithStats(bbox_mask, connectivity=8)
    if n_bb <= 1:
        return _extract_tumor_region_hybrid(cam, original_img)
    idx_bb = 1 + int(np.argmax(stats_bb[1:, cv2.CC_STAT_AREA]))
    cam_bbox = {
        'x': int(stats_bb[idx_bb, cv2.CC_STAT_LEFT]),
        'y': int(stats_bb[idx_bb, cv2.CC_STAT_TOP]),
        'w': int(stats_bb[idx_bb, cv2.CC_STAT_WIDTH]),
        'h': int(stats_bb[idx_bb, cv2.CC_STAT_HEIGHT]),
    }

    # SAM segmentation
    pil_224 = original_img.resize((224, 224))
    sam_mask = _sam_segment_mask(pil_224, cam_bbox)
    if sam_mask is not None:
        sam_mask = cv2.bitwise_and(sam_mask, brain_mask)
        n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(sam_mask, connectivity=8)
        if n_cc > 1:
            i_cc = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
            largest = (lbl_cc == i_cc).astype(np.uint8) * 255
            tumor_area = int((largest > 0).sum())
            size_pct = tumor_area / float(brain_area)
            if size_pct < 0.40:
                ys, xs = np.where(largest > 0)
                refined_bbox = {
                    'x': int(xs.min()), 'y': int(ys.min()),
                    'w': int(xs.max() - xs.min() + 1),
                    'h': int(ys.max() - ys.min() + 1),
                }
                _ok, png = cv2.imencode('.png', largest)
                mask_b64 = base64.b64encode(png.tobytes()).decode('utf-8') if _ok else None
                return {
                    'size_pct': size_pct,
                    'bbox': refined_bbox,
                    'brain_area': brain_area,
                    'tumor_area': tumor_area,
                    'method': 'sam_from_cam_bbox',
                    'mask_b64': mask_b64,
                    'convention': 'radiological',
                }

    return _extract_tumor_region_hybrid(cam, original_img)


def _extract_tumor_region_hybrid(cam, original_img):
    """Hybrid CAM + intensity segmentation for accurate tumor sizing.

    Pure Grad-CAM++ overestimates (shows full attention region including edema).
    Pure K-means overestimates (selects entire intensity cluster as "tumor").

    This approach combines both signals:
      1. CAM → rough ROI (85th percentile bbox for localisation)
      2. Within ROI, compute CAM-weighted intensity anomaly score per pixel
      3. Otsu threshold on that score → tight tumor mask
      4. Fallback: 95th percentile CAM threshold if Otsu produces outliers

    The key insight: tumor pixels have BOTH high CAM activation AND
    anomalous intensity (different from surrounding normal brain).
    Normal brain with high CAM (context features) gets filtered out
    because its intensity is typical, not anomalous.
    """
    cam_r = cv2.resize(cam, (224, 224))
    orig = np.array(original_img.resize((224, 224)))
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY) if orig.ndim == 3 else orig
    _, brain_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_close = np.ones((5, 5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel_close)
    brain_area = int((brain_mask > 0).sum())
    if brain_area == 0:
        return {'size_pct': 0.0, 'bbox': None, 'brain_area': 0, 'method': 'none'}
    pos = cam_r[cam_r > 0]
    if pos.size == 0:
        return {'size_pct': 0.0, 'bbox': None, 'brain_area': brain_area, 'method': 'none'}

    # ── Step 1: CAM-based bbox for ROI localisation (85th pct + erosion) ──
    thresh_bbox = float(np.percentile(pos, 85))
    bbox_mask = cv2.bitwise_and((cam_r >= thresh_bbox).astype(np.uint8) * 255, brain_mask)
    kernel_erode = np.ones((3, 3), np.uint8)
    bbox_mask = cv2.erode(bbox_mask, kernel_erode, iterations=1)
    n_bb, _, stats_bb, _ = cv2.connectedComponentsWithStats(bbox_mask, connectivity=8)
    if n_bb <= 1:
        return {'size_pct': 0.0, 'bbox': None, 'brain_area': brain_area, 'method': 'none'}
    idx_bb = 1 + int(np.argmax(stats_bb[1:, cv2.CC_STAT_AREA]))
    bx = int(stats_bb[idx_bb, cv2.CC_STAT_LEFT])
    by = int(stats_bb[idx_bb, cv2.CC_STAT_TOP])
    bw = int(stats_bb[idx_bb, cv2.CC_STAT_WIDTH])
    bh = int(stats_bb[idx_bb, cv2.CC_STAT_HEIGHT])

    # ── Step 2: CAM-weighted intensity anomaly within ROI ──
    pad = 0.15
    rx1 = max(0, int(bx - bw * pad))
    ry1 = max(0, int(by - bh * pad))
    rx2 = min(224, int(bx + bw + bw * pad))
    ry2 = min(224, int(by + bh + bh * pad))
    roi_gray = gray[ry1:ry2, rx1:rx2].astype(np.float32)
    roi_cam = cam_r[ry1:ry2, rx1:rx2].astype(np.float32)
    roi_brain = brain_mask[ry1:ry2, rx1:rx2]

    brain_pix = roi_brain > 0
    if brain_pix.sum() < 50:
        cam_area = int(stats_bb[idx_bb, cv2.CC_STAT_AREA])
        return {
            'size_pct': float(cam_area) / float(brain_area),
            'bbox': {'x': bx, 'y': by, 'w': bw, 'h': bh},
            'brain_area': brain_area, 'tumor_area': cam_area,
            'method': 'cam_fallback',
        }

    # Compute intensity anomaly: how different each pixel is from the local median
    brain_intensities = roi_gray[brain_pix]
    median_intensity = float(np.median(brain_intensities))
    intensity_dev = np.abs(roi_gray - median_intensity)
    # Normalize both signals to [0, 1]
    id_max = intensity_dev.max()
    if id_max > 0:
        intensity_dev = intensity_dev / id_max
    cam_max = roi_cam.max()
    if cam_max > 0:
        roi_cam_norm = roi_cam / cam_max
    else:
        roi_cam_norm = roi_cam

    # Combined score: multiplicative — only pixels with BOTH high CAM
    # AND high intensity anomaly survive. This eliminates normal brain
    # that happens to have high CAM (model context features).
    combined = (roi_cam_norm * intensity_dev) * (brain_pix.astype(np.float32))

    # Otsu on the combined score
    combined_u8 = (combined * 255).astype(np.uint8)
    _, tumor_mask = cv2.threshold(combined_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tumor_mask = cv2.bitwise_and(tumor_mask, roi_brain)

    # Morphological cleanup: close gaps, then open to remove noise
    k_morph = np.ones((3, 3), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, k_morph)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, k_morph)

    # Find largest connected component
    n_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(tumor_mask, connectivity=8)
    if n_cc <= 1:
        # Fallback: tight 95th pct CAM threshold
        t95 = float(np.percentile(pos, 95))
        fb_mask = cv2.bitwise_and((cam_r >= t95).astype(np.uint8) * 255, brain_mask)
        fb_mask = cv2.erode(fb_mask, kernel_erode, iterations=1)
        n3, _, s3, _ = cv2.connectedComponentsWithStats(fb_mask, connectivity=8)
        if n3 <= 1:
            return {'size_pct': 0.0, 'bbox': {'x': bx, 'y': by, 'w': bw, 'h': bh},
                    'brain_area': brain_area, 'tumor_area': 0, 'method': 'cam_95_fallback'}
        i3 = 1 + int(np.argmax(s3[1:, cv2.CC_STAT_AREA]))
        ta = int(s3[i3, cv2.CC_STAT_AREA])
        return {
            'size_pct': float(ta) / float(brain_area),
            'bbox': {'x': int(s3[i3, cv2.CC_STAT_LEFT]), 'y': int(s3[i3, cv2.CC_STAT_TOP]),
                     'w': int(s3[i3, cv2.CC_STAT_WIDTH]), 'h': int(s3[i3, cv2.CC_STAT_HEIGHT])},
            'brain_area': brain_area, 'tumor_area': ta, 'method': 'cam_95_fallback',
        }

    idx_cc = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
    tumor_area = int(stats_cc[idx_cc, cv2.CC_STAT_AREA])
    size_pct = float(tumor_area) / float(brain_area)

    # Sanity check: if combined method gives >25%, fall back to tight CAM
    if size_pct > 0.25:
        t95 = float(np.percentile(pos, 95))
        fb_mask = cv2.bitwise_and((cam_r >= t95).astype(np.uint8) * 255, brain_mask)
        fb_mask = cv2.erode(fb_mask, kernel_erode, iterations=1)
        n3, _, s3, _ = cv2.connectedComponentsWithStats(fb_mask, connectivity=8)
        if n3 > 1:
            i3 = 1 + int(np.argmax(s3[1:, cv2.CC_STAT_AREA]))
            tumor_area = int(s3[i3, cv2.CC_STAT_AREA])
            size_pct = float(tumor_area) / float(brain_area)
            return {
                'size_pct': size_pct,
                'bbox': {'x': int(s3[i3, cv2.CC_STAT_LEFT]), 'y': int(s3[i3, cv2.CC_STAT_TOP]),
                         'w': int(s3[i3, cv2.CC_STAT_WIDTH]), 'h': int(s3[i3, cv2.CC_STAT_HEIGHT])},
                'brain_area': brain_area, 'tumor_area': tumor_area,
                'method': 'cam_95_capped',
                'convention': 'radiological',
            }

    # Convert component bbox back to full 224x224 coordinates
    cx_roi = int(stats_cc[idx_cc, cv2.CC_STAT_LEFT])
    cy_roi = int(stats_cc[idx_cc, cv2.CC_STAT_TOP])
    cw = int(stats_cc[idx_cc, cv2.CC_STAT_WIDTH])
    ch = int(stats_cc[idx_cc, cv2.CC_STAT_HEIGHT])
    final_x = rx1 + cx_roi
    final_y = ry1 + cy_roi

    return {
        'size_pct': size_pct,
        'bbox': {'x': final_x, 'y': final_y, 'w': cw, 'h': ch},
        'brain_area': brain_area,
        'tumor_area': tumor_area,
        'method': 'hybrid_cam_intensity',
        'convention': 'radiological',
    }


# 3×3 grid over the 224×224 model-input axial slice.
# x: 0=image-left, 224=image-right.  y: 0=anterior (top), 224=posterior (bottom).
# Note on sides: image-left ≈ patient-right (radiology convention), but datasets vary,
# so we report image-relative sides only.
_REGION_GRID = {
    (0, 0): ('Right frontal',                             'right',   'anterior'),
    (1, 0): ('Anterior midline',                          'midline', 'anterior'),
    (2, 0): ('Left frontal',                              'left',    'anterior'),
    (0, 1): ('Right temporal/parietal',                   'right',   'middle'),
    (1, 1): ('Deep central (thalamic / sellar)',          'midline', 'middle'),
    (2, 1): ('Left temporal/parietal',                    'left',    'middle'),
    (0, 2): ('Right occipital',                           'right',   'posterior'),
    (1, 2): ('Posterior midline (cerebellum / brainstem)', 'midline', 'posterior'),
    (2, 2): ('Left occipital',                            'left',    'posterior'),
}

_REGION_GRID_ES = {
    (0, 0): ('Frontal derecho',                                   'right',   'anterior'),
    (1, 0): ('Línea media anterior',                              'midline', 'anterior'),
    (2, 0): ('Frontal izquierdo',                                 'left',    'anterior'),
    (0, 1): ('Temporal/parietal derecho',                         'right',   'middle'),
    (1, 1): ('Central profundo (talámico / sellar)',              'midline', 'middle'),
    (2, 1): ('Temporal/parietal izquierdo',                       'left',    'middle'),
    (0, 2): ('Occipital derecho',                                 'right',   'posterior'),
    (1, 2): ('Línea media posterior (cerebelo / tronco encefálico)', 'midline', 'posterior'),
    (2, 2): ('Occipital izquierdo',                               'left',    'posterior'),
}

# Clinical priors: which 3×3 cells are typical / atypical for each tumor type.
# Sources: standard neuroradiology references (gliomas favor cerebral hemispheres,
# meningiomas arise from dura → convexity / parasagittal / falx, pituitary → sella turcica).
_CLINICAL_PRIORS = {
    'Glioma': {
        'typical': {(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)},
        'atypical': {(1, 0), (1, 1), (1, 2)},
        'note_typical': 'Gliomas most commonly arise in the cerebral hemispheres (frontal/temporal/parietal). Attention region is consistent with this prior.',
        'note_atypical': 'Midline gliomas (brainstem, thalamic, diffuse midline) do occur but are less common. Recommend review.',
    },
    'Meningioma': {
        'typical': {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)},
        'atypical': {(0, 1), (2, 1), (0, 2), (2, 2)},
        'note_typical': 'Meningiomas typically arise from dural surfaces (parasagittal, convexity, falx, posterior midline). Consistent with this prior.',
        'note_atypical': 'Meningiomas are rarely deep / intraparenchymal. A meningioma label with attention deep in the cerebrum is atypical — recommend review.',
    },
    'Pituitary': {
        'typical': {(1, 1)},
        'atypical': {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)},
        'note_typical': 'Pituitary adenomas arise in the sella turcica (deep central). Consistent with this prior.',
        'note_atypical': 'Pituitary tumors occupy the sella turcica. Attention outside the deep central region is atypical — recommend review.',
    },
}

_CLINICAL_PRIORS_ES_NOTES = {
    'Glioma': {
        'note_typical': 'Los gliomas suelen surgir en los hemisferios cerebrales (frontal/temporal/parietal). La región de atención es consistente con este patrón.',
        'note_atypical': 'Los gliomas de línea media (tronco encefálico, talámico, difuso de línea media) existen pero son menos comunes. Se recomienda revisión.',
    },
    'Meningioma': {
        'note_typical': 'Los meningiomas suelen surgir de superficies durales (parasagital, convexidad, hoz, línea media posterior). Consistente con este patrón.',
        'note_atypical': 'Los meningiomas rara vez son profundos / intraparenquimatosos. Una etiqueta de meningioma con atención en zona cerebral profunda es atípica — se recomienda revisión.',
    },
    'Pituitary': {
        'note_typical': 'Los adenomas hipofisarios surgen en la silla turca (central profundo). Consistente con este patrón.',
        'note_atypical': 'Los tumores hipofisarios ocupan la silla turca. La atención fuera de la región central profunda es atípica — se recomienda revisión.',
    },
}


def compute_anatomical_region(bbox, class_name, img_size=224, lang='en'):
    """Map Grad-CAM++ bbox centroid → anatomical region + clinical-prior consistency check."""
    if not bbox or class_name == 'No Tumor' or class_name not in _CLINICAL_PRIORS:
        return None
    cx = bbox['x'] + bbox['w'] / 2.0
    cy = bbox['y'] + bbox['h'] / 2.0
    third = img_size / 3.0
    xb = 0 if cx < third else (1 if cx < 2 * third else 2)
    yb = 0 if cy < third else (1 if cy < 2 * third else 2)
    grid = _REGION_GRID_ES if pick_lang(lang) == 'es' else _REGION_GRID
    label, side, zone = grid[(xb, yb)]
    prior = _CLINICAL_PRIORS[class_name]
    notes_es = _CLINICAL_PRIORS_ES_NOTES.get(class_name) if pick_lang(lang) == 'es' else None
    if (xb, yb) in prior['typical']:
        consistency = 'typical'
        explanation = (notes_es or prior)['note_typical']
    elif (xb, yb) in prior['atypical']:
        consistency = 'atypical'
        explanation = (notes_es or prior)['note_atypical']
    else:
        consistency, explanation = 'unknown', ''
    disclaimer = (
        'Región aproximada por el centroide de la bbox Grad-CAM++ sobre la entrada 224×224. No sustituye la segmentación de un radiólogo.'
        if pick_lang(lang) == 'es' else
        'Region approximated from Grad-CAM++ bbox centroid on 224×224 model input. Not a substitute for radiologist segmentation.'
    )
    return {
        'label': label,
        'side': side,
        'axial_zone': zone,
        'centroid': {'x': int(cx), 'y': int(cy)},
        'consistency': consistency,
        'explanation': explanation,
        'disclaimer': disclaimer,
    }


def compute_malignancy(class_name, confidence, cam_pp, original_img, lang='en'):
    base = get_base_risk(class_name, lang)
    if class_name == 'No Tumor':
        return {
            'score': 0.0,
            'score_out_of_10': '0.0/10',
            'base_risk': base['label'],
            'base_score': base['score'],
            'clinical_note': base['note'],
            'size_pct': 0.0,
            'bbox': None,
            'summary': ('No se detectó tumor — la malignidad no es aplicable.'
                        if pick_lang(lang) == 'es' else
                        'No tumor detected — malignancy not applicable.'),
        }
    region = _extract_tumor_region(cam_pp, original_img)
    size_pct = region['size_pct']
    seg_method = region.get('method', 'unknown')

    size_bonus = min(size_pct / 0.3, 1.0) * 2.0
    raw = base['score'] + size_bonus
    final = raw * (0.6 + 0.4 * float(confidence))
    final = round(max(0.0, min(10.0, final)), 1)
    size_category = 'small' if size_pct < 0.05 else ('medium' if size_pct < 0.15 else 'large')
    anatomical = compute_anatomical_region(region['bbox'], class_name, lang=lang)
    # Human-readable method label — distinguish YOLO/SAM/CAM paths.
    if seg_method in ('yolo_sam', 'yolo_bbox', 'yolo_bbox_sam_overlay', 'yolo_only'):
        suffix = ' + SAM mask overlay' if seg_method == 'yolo_bbox_sam_overlay' else ''
        size_method_label = (
            f"YOLO detector (Cheng 2015, mAP50=0.92) bbox{suffix} — "
            f"conf {region.get('yolo_conf', 0):.2f}"
        )
    elif seg_method == 'sam_from_cam_bbox':
        size_method_label = 'SAM (MobileSAM) prompted by Grad-CAM++ bbox — pixel-tight segmentation'
    else:
        size_method_label = f'Hybrid CAM-guided intensity segmentation ({seg_method})'
    # Attach per-class clinical context (static medical knowledge, language-aware)
    ctx = get_clinical_context(class_name, lang)
    clinical_context = None
    if ctx:
        clinical_context = {
            'primer':           ctx['primer'],
            'size_interpretation': ctx['size_interpretation'].get(size_category),
            'subtypes':         ctx['subtypes'],
            'typical_workup':   ctx['typical_workup'],
        }
    # Urgency banner (deterministic, no LLM) + medical coding suggestions
    urgency = compute_urgency(class_name, final, size_pct * 100, lang=lang)
    medical_codes = get_medical_codes(class_name, lang=lang)
    size_pct_pct = round(size_pct * 100, 1)
    size_range = size_pct_to_range(size_pct_pct)
    size_range_label = size_range[2] if size_range else 'unknown'
    print(f'[SIZE] class={class_name} '
          f'yolo_bbox_area={region.get("tumor_area")} '
          f'brain_area={region.get("brain_area")} '
          f'pixel_pct={size_pct_pct} '
          f'range={size_range_label} '
          f'method={seg_method}', flush=True)
    return {
        'score': final,
        'score_out_of_10': f'{final}/10',
        'base_risk': base['label'],
        'base_score': base['score'],
        'clinical_note': base['note'],
        'size_pct': size_pct_pct,
        'size_range': size_range_label,
        'size_range_low':  size_range[0] if size_range else None,
        'size_range_high': size_range[1] if size_range else None,
        'size_category': size_category,
        'size_method': size_method_label,
        'size_debug': {
            'tumor_area':      region.get('tumor_area'),
            'tumor_area_orig': region.get('tumor_area_orig'),
            'brain_area':      region.get('brain_area'),
            'brain_area_orig': region.get('brain_area_orig'),
            'bbox_area':       region.get('bbox_area'),
            'sam_area':        region.get('sam_area'),
            'sam_ratio':       region.get('sam_ratio'),
            'size_pct_224':    region.get('size_pct_224'),
            'size_pct_orig':   region.get('size_pct_orig'),
            'method':          region.get('method'),
        },
        'bbox': region['bbox'],
        'bbox_source': region.get('bbox_source'),
        'yolo_conf': region.get('yolo_conf'),
        'mask_b64': region.get('mask_b64'),
        'region': anatomical,
        'clinical_context': clinical_context,
        'urgency': urgency,
        'medical_codes': medical_codes,
        'convention': 'radiological (patient left = image right)',
        'summary': f"Type: {class_name} | Base risk: {base['label']} | ~{size_range_label} (est.) | Malignancy: {final}/10",
    }




# ── Novel #5: Adversarial Robustness Testing ───────────────────────
def test_robustness(model, model_name, image_tensor, device, target_class):
    model.eval()
    image_tensor = image_tensor.to(device)
    tl = get_target_layer(model, model_name)
    clean_cam = cv2.resize(_gradcam(model, image_tensor, tl, target_class), (224, 224))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_01 = torch.clamp(image_tensor.cpu() * std + mean, 0, 1)

    def renorm(t):
        return ((t - mean) / std).to(device)

    def blur_t(t):
        arr = t.squeeze(0).numpy()
        bl = np.stack([gaussian_filter(arr[c], sigma=1.0) for c in range(3)])
        return torch.from_numpy(bl).unsqueeze(0).float()

    def rician(t, sigma=0.03):
        r = t + torch.randn_like(t) * sigma
        i_noise = torch.randn_like(t) * sigma
        return torch.clamp(torch.sqrt(r ** 2 + i_noise ** 2), 0, 1)

    perts = {
        'gaussian_noise': torch.clamp(img_01 + torch.randn_like(img_01) * 0.03, 0, 1),
        'brightness': torch.clamp(img_01 * 1.2, 0, 1),
        'scanner_noise': rician(img_01),
        'blur': blur_t(img_01),
    }
    with torch.no_grad():
        clean_pred = model(image_tensor).argmax(1).item()
    results = {}
    scores = []
    for name, pert_01 in perts.items():
        pert_norm = renorm(pert_01)
        with torch.no_grad():
            pert_pred = model(pert_norm).argmax(1).item()
        pert_cam = cv2.resize(_gradcam(model, pert_norm, tl, target_class), (224, 224))
        a, b = clean_cam.flatten(), pert_cam.flatten()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        xai_stab = float(np.dot(a, b) / d) if d > 1e-8 else 0.0
        pred_ok = clean_pred == pert_pred
        combined = 0.5 * (1.0 if pred_ok else 0.0) + 0.5 * xai_stab
        scores.append(combined)
        results[name] = {
            'pred_stable': bool(pred_ok),
            'xai_stability': round(xai_stab, 4),
            'combined': round(combined, 4)
        }
    overall = round(float(np.mean(scores)), 4)
    return {'overall_score': overall, 'fda_ready': overall >= 0.85, 'tests': results}


# ── Test-Time Augmentation (TTA) ────────────────────────────────────
# Generate N slightly-perturbed copies of the preprocessed tensor and average
# softmax outputs across them. Improves robustness to image-quality variation
# (brightness, contrast, scanner artifacts, small rotations) without any retrain.
TTA_OPS = ['original', 'hflip', 'rotate+5', 'rotate-5', 'brightness+']


def _tta_versions(image_tensor):
    """Return list of (op_name, augmented_tensor) for TTA averaging."""
    import torchvision.transforms.functional as TF
    return [
        ('original',     image_tensor),
        ('hflip',        torch.flip(image_tensor, dims=[3])),
        ('rotate+5',     TF.rotate(image_tensor, 5)),
        ('rotate-5',     TF.rotate(image_tensor, -5)),
        ('brightness+',  torch.clamp(image_tensor * 1.05, -3.0, 3.0)),
    ]


def _tta_predict(model, tta_tensors, device):
    """Run model on every TTA variant and return mean softmax probabilities."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for _name, t in tta_tensors:
            logits = model(t.to(device))
            all_probs.append(F.softmax(logits, dim=1).cpu().numpy()[0])
    return np.mean(np.stack(all_probs), axis=0)


# ── API Endpoints ──────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty file uploaded'}), 400
        if len(image_bytes) > 20 * 1024 * 1024:
            return jsonify({'error': 'File too large (max 20MB)'}), 400

        all_models, device = get_models()
        # Read preprocessing mode from form field (default: standard)
        cross_scanner = request.form.get('preprocessing_mode', 'standard') == 'cross_scanner'
        lang = request.form.get('language', 'en')
        try:
            image_tensor, original_img, prep_info = preprocess_image(image_bytes, cross_scanner=cross_scanner)
        except ValueError as ve:
            return jsonify({'error': str(ve)}), 400
        image_tensor = image_tensor.to(device)

        # Pre-compute the TTA variants once, then reuse for every model.
        tta_variants = _tta_versions(image_tensor)

        # Run each model — TTA-average the softmax for the headline confidence,
        # keep the un-augmented logits for the energy-based OOD score.
        model_results = {}
        model_energies = {}
        for mname, model in all_models.items():
            model.eval()
            # 1. Single deterministic pass for the OOD energy score
            with torch.no_grad():
                logits = model(image_tensor)
                model_energies[mname] = float(-torch.logsumexp(logits, dim=1).item())
            # 2. TTA-averaged probabilities for the prediction class + confidence
            probs = _tta_predict(model, tta_variants, device)
            pred_class = int(probs.argmax())
            model_results[mname] = {
                'class': CLASS_NAMES[pred_class],
                'confidence': round(float(probs[pred_class]), 6),
                'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(4)},
                'pred_index': pred_class,
                'tta_applied': True,
            }

        best_name = max(model_results, key=lambda k: model_results[k]['confidence'])
        best_model = all_models[best_name]

        uncertainty = predict_with_uncertainty(best_model, image_tensor, device, T=20)
        target_class = uncertainty['prediction']
        xai, raw_cams = hierarchical_xai(best_model, best_name, image_tensor, device, target_class, original_img)
        robustness = test_robustness(best_model, best_name, image_tensor, device, target_class)
        malignancy = compute_malignancy(
            class_name=uncertainty['class_name'],
            confidence=uncertainty['mean_confidence'],
            cam_pp=raw_cams['gradcam_pp'],
            original_img=original_img,
            lang=lang,
        )

        # Always produce the zoomed tumor crop with bbox drawn — decoupled from
        # the optional Focus-Crop self-check. The malignancy card needs this image
        # whenever there's a tumor bbox available, regardless of whether the focus
        # classifier re-check ran.
        if malignancy and malignancy.get('bbox') and uncertainty['class_name'] != 'No Tumor':
            try:
                disp_pil, _disp_origin, _tumor_in_crop = display_crop_with_bbox(
                    original_img, malignancy['bbox'], padding_frac=1.0, min_side=240,
                )
                if disp_pil is not None:
                    _buf = io.BytesIO()
                    disp_pil.save(_buf, format='PNG')
                    malignancy['tumor_crop_image'] = base64.b64encode(_buf.getvalue()).decode('utf-8')
                    malignancy['tumor_crop_size'] = list(disp_pil.size)
            except Exception as _crop_e:
                print(f'[tumor-crop] {_crop_e}', flush=True)

        if robustness['overall_score'] < 0.50:
            uncertainty['needs_review'] = True

        # ── Tumor-side OOD detection (energy score on best_model logits) ──
        tumor_ood_block = None
        try:
            tumor_cal = get_tumor_ood_calibration()
            cal_for_model = tumor_cal.get(best_name) if isinstance(tumor_cal, dict) else None
            best_energy = model_energies.get(best_name)
            if cal_for_model and best_energy is not None:
                threshold = float(cal_for_model.get('threshold_used') or cal_for_model.get('p95'))
                is_ood_tumor = bool(best_energy > threshold)
                tumor_ood_block = {
                    'method': 'Energy-based OOD detection (Liu et al. 2020): E(x) = -logsumexp(logits)',
                    'best_model': best_name,
                    'energy_score': round(best_energy, 4),
                    'threshold_p95': round(threshold, 4),
                    'p99_threshold': round(float(cal_for_model.get('p99', threshold)), 4),
                    'val_mean': round(float(cal_for_model.get('mean', 0)), 4),
                    'is_ood': is_ood_tumor,
                    'note': ('Energy beyond the in-distribution 95th percentile suggests this scan '
                             'is outside the training distribution — predictions including "No Tumor" '
                             'should be interpreted with caution regardless of softmax confidence.'),
                }
                # Stricter needs_review on the tumor side too
                probs_sorted = sorted(uncertainty['probabilities'].values(), reverse=True)
                top2_gap = float(probs_sorted[0] - probs_sorted[1]) if len(probs_sorted) >= 2 else 1.0
                stricter_review = (
                    bool(uncertainty['needs_review'])
                    or uncertainty['epistemic'] > 0.05
                    or uncertainty['mean_confidence'] < 0.85
                    or top2_gap < 0.30
                    or is_ood_tumor
                )
                uncertainty['needs_review'] = bool(stricter_review)
                uncertainty['top2_gap'] = round(top2_gap, 4)
        except Exception as _ood_e:
            import traceback; traceback.print_exc()
            tumor_ood_block = {'method': 'energy', 'enabled': False, 'error': str(_ood_e)}

        # ── Focus-Crop self-check: re-classify on the Grad-CAM++ tumor bbox ─
        # Pass 1: classifier saw the whole image (already done above, lives in `uncertainty`).
        # Pass 2: crop to the model's own attention region, re-classify. Agreement = the
        # model is genuinely focused on the tumor; disagreement signals reliance on
        # non-tumor (background / scanner) features and is treated as a domain-shift hint.
        focus_crop_block = None
        bbox_224 = malignancy.get('bbox') if malignancy else None
        if uncertainty['class_name'] != 'No Tumor' and bbox_224:
            try:
                cropped_pil, bbox_orig = crop_to_tumor(original_img, bbox_224, padding_frac=0.05)
                if cropped_pil is not None:
                    crop_tensor, _crop_pil, _crop_prep = _preprocess_pil(cropped_pil)
                    crop_tensor = crop_tensor.to(device)
                    best_model.eval()
                    with torch.no_grad():
                        crop_logits = best_model(crop_tensor)
                        crop_probs = F.softmax(crop_logits, dim=1).cpu().numpy()[0]
                    crop_pred = int(crop_probs.argmax())
                    crop_class = CLASS_NAMES[crop_pred]
                    crop_conf = float(crop_probs[crop_pred])

                    full_class = uncertainty['class_name']
                    full_conf = float(uncertainty['mean_confidence'])
                    class_match = (crop_class == full_class)
                    conf_delta = crop_conf - full_conf
                    # consistency rule: same class AND confidence didn't drop by more than 0.15
                    is_consistent = bool(class_match and conf_delta > -0.15)
                    if class_match:
                        verdict = 'consistent' if conf_delta > -0.15 else 'confidence_dropped'
                    else:
                        verdict = 'class_changed'

                    # Build the *display* crop for the UI: wider context + drawn
                    # bbox showing where the tumor is. Falls back to the tight
                    # focus-classifier crop if the wide variant can't be made.
                    display_pil, _disp_origin, _tumor_in_crop = display_crop_with_bbox(
                        original_img, bbox_224, padding_frac=1.0, min_side=240,
                    )
                    if display_pil is None:
                        display_pil = cropped_pil
                    buf = io.BytesIO()
                    display_pil.save(buf, format='PNG')
                    crop_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                    focus_crop_block = {
                        'enabled': True,
                        'method': 'Grad-CAM++ guided self-attention crop and re-classify',
                        'description': ('Pass 1 sees the whole image; Pass 2 sees only the bbox the model itself attended to. '
                                        'Agreement = model is genuinely focused on the tumor. Disagreement = model relied on '
                                        'non-tumor features (likely domain shift).'),
                        'padding_frac': 0.15,
                        'bbox_224': bbox_224,
                        'bbox_orig': bbox_orig,
                        'crop_image': crop_b64,
                        'crop_size': list(display_pil.size),
                        'full_prediction': {
                            'class': full_class,
                            'confidence': round(full_conf, 4),
                        },
                        'crop_prediction': {
                            'class': crop_class,
                            'confidence': round(crop_conf, 4),
                            'probabilities': {CLASS_NAMES[i]: round(float(crop_probs[i]), 4) for i in range(4)},
                        },
                        'agreement': {
                            'class_match': bool(class_match),
                            'confidence_delta': round(conf_delta, 4),
                            'is_consistent': is_consistent,
                            'verdict': verdict,
                        },
                    }
                    if not is_consistent:
                        uncertainty['needs_review'] = True
                else:
                    focus_crop_block = {
                        'enabled': False,
                        'reason': 'tumor bbox too small to crop (likely weak Grad-CAM++ activation)'
                    }
            except Exception as fc_e:
                import traceback
                traceback.print_exc()
                focus_crop_block = {'enabled': False, 'error': str(fc_e)}

        # ── MedGemma multimodal tumor assessment ──
        medgemma_tumor = None
        if malignancy and uncertainty['class_name'] != 'No Tumor':
            try:
                from llm.medgemma_client import assess_tumor as _assess_tumor
                # Convert original image to base64 for multimodal input
                _img_buf = io.BytesIO()
                original_img.save(_img_buf, format='PNG')
                _img_b64 = base64.b64encode(_img_buf.getvalue()).decode('utf-8')
                medgemma_tumor = _assess_tumor(
                    image_b64=_img_b64,
                    class_name=uncertainty['class_name'],
                    confidence=uncertainty['mean_confidence'],
                    bbox_224=malignancy.get('bbox'),
                    size_pct=malignancy.get('size_pct', 0.0),
                    language=lang,
                )
                if medgemma_tumor.get('success') and medgemma_tumor.get('assessment'):
                    malignancy['medgemma_assessment'] = medgemma_tumor['assessment']
                    malignancy['medgemma_model'] = medgemma_tumor.get('model')
                    malignancy['medgemma_duration_ms'] = medgemma_tumor.get('total_duration_ms')

                    # ── If MedGemma gave us a normalized bbox, prefer it as the
                    # SAM prompt over Grad-CAM++. Grad-CAM++ shows what the
                    # classifier looked at, which is often *not* the tumor itself.
                    # MedGemma actually visually localizes the lesion.
                    mg = medgemma_tumor['assessment'] or {}
                    mg_bbox_sam_ok = False
                    mg_bbox_sam_size_pct = None
                    # If YOLO already produced a confident bbox in
                    # _extract_tumor_region, don't let MedGemma override it —
                    # the trained detector is more reliable than MedGemma's
                    # visual-language spatial reasoning.
                    yolo_already_won = (malignancy.get('bbox_source') == 'yolo')
                    bbox_norm = None if yolo_already_won else mg.get('bbox_norm')
                    if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
                        try:
                            bx, by, bw, bh = [float(v) for v in bbox_norm]
                            ok = (0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0
                                  and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0
                                  and bx + bw <= 1.001 and by + bh <= 1.001)
                        except (TypeError, ValueError):
                            ok = False
                        if ok:
                            mg_bbox_224 = {
                                'x': max(0, int(round(bx * 224))),
                                'y': max(0, int(round(by * 224))),
                                'w': max(8, int(round(bw * 224))),
                                'h': max(8, int(round(bh * 224))),
                            }
                            mg_bbox_224['w'] = min(mg_bbox_224['w'], 224 - mg_bbox_224['x'])
                            mg_bbox_224['h'] = min(mg_bbox_224['h'], 224 - mg_bbox_224['y'])

                            # Run multi-strategy SAM with MedGemma's bbox as the seed.
                            # Tries 4 prompt variants (bbox, expanded bbox, peak-intensity
                            # point, bbox+point) and picks the candidate with highest
                            # intensity contrast against surrounding brain — much tighter
                            # masks for enhancing lesions than single-bbox SAM.
                            try:
                                pil_224 = original_img.resize((224, 224))
                                gray = cv2.cvtColor(np.array(pil_224.convert('RGB')), cv2.COLOR_RGB2GRAY)
                                brain_mask = _real_brain_mask(gray)
                                brain_area = int((brain_mask > 0).sum())
                                # Build the extra-seeds list:
                                #   (a) Top-3 most-intense Grad-CAM++ peaks — replaces the
                                #       single "largest connected component" seed, which lost
                                #       small-but-intense tumor activations to larger weaker
                                #       background blobs.
                                #   (b) The legacy SAM-refined cam_seed (from _extract_tumor_region)
                                #       kept as a safety net.
                                #   (c) Class-aware Pituitary prior (sella turcica).
                                extra_seeds = []
                                try:
                                    extra_seeds.extend(_cam_top_k_bboxes(
                                        raw_cams['gradcam_pp'], brain_mask, k=3,
                                    ))
                                except Exception as _e_topk:
                                    print(f'[cam-topk] {_e_topk}', flush=True)
                                cam_seed = malignancy.get('bbox') if malignancy else None
                                if cam_seed:
                                    extra_seeds.append(cam_seed)
                                # Class-aware anatomical prior — for Pituitary, the
                                # tumor is anatomically constrained to the sella turcica
                                # (geometric center / slightly anterior on axial slices).
                                _cls = (uncertainty.get('class_name') or '').strip()
                                if _cls == 'Pituitary':
                                    prior_bbox = {
                                        'x': int(0.375 * 224), 'y': int(0.30 * 224),
                                        'w': int(0.25 * 224),  'h': int(0.28 * 224),
                                    }
                                    extra_seeds.append(prior_bbox)
                                largest, sam_strategy = _sam_segment_refined(
                                    pil_224, mg_bbox_224,
                                    brain_mask=brain_mask,
                                    extra_bboxes=extra_seeds if extra_seeds else None,
                                )
                                if largest is not None and brain_area > 0:
                                    tumor_area = int((largest > 0).sum())
                                    new_size_pct = tumor_area / float(brain_area)
                                    if 0.001 < new_size_pct < 0.50:
                                        # Stash the MG-bbox-SAM measurement as a
                                        # comparison metric; do NOT overwrite the
                                        # YOLO-derived bbox/mask if YOLO was the
                                        # source (YOLO bbox is supervised and more
                                        # reliable for display).
                                        malignancy['pixel_size_pct_sam_medgemma'] = round(new_size_pct * 100, 1)
                                        malignancy['sam_strategy'] = sam_strategy
                                        mg_bbox_sam_ok = True
                                        mg_bbox_sam_size_pct = round(new_size_pct * 100, 1)
                                        if malignancy.get('bbox_source') != 'yolo':
                                            # YOLO didn't fire — replace the bbox/mask
                                            # with the MG-bbox-SAM result.
                                            ys, xs = np.where(largest > 0)
                                            new_bbox = {
                                                'x': int(xs.min()), 'y': int(ys.min()),
                                                'w': int(xs.max() - xs.min() + 1),
                                                'h': int(ys.max() - ys.min() + 1),
                                            }
                                            _ok_png, _png = cv2.imencode('.png', largest)
                                            new_mask_b64 = base64.b64encode(_png.tobytes()).decode('utf-8') if _ok_png else None
                                            malignancy['bbox'] = new_bbox
                                            malignancy['mask_b64'] = new_mask_b64
                                            malignancy['size_source'] = 'medgemma_bbox_sam'
                                            malignancy['bbox_source'] = 'medgemma'
                                            # Regenerate the zoomed crop with the new bbox
                                            disp_pil, _o, _t = display_crop_with_bbox(
                                                original_img, new_bbox,
                                                padding_frac=1.0, min_side=240,
                                            )
                                            if disp_pil is not None:
                                                _buf = io.BytesIO()
                                                disp_pil.save(_buf, format='PNG')
                                                malignancy['tumor_crop_image'] = base64.b64encode(_buf.getvalue()).decode('utf-8')
                                                malignancy['tumor_crop_size'] = list(disp_pil.size)
                            except Exception as _sam_e:
                                print(f'[medgemma-sam] {_sam_e}', flush=True)

                    # Stash the original SAM/pixel-derived values for debugging.
                    # IMPORTANT: do NOT overwrite size_source if the MedGemma-bbox SAM
                    # path already won — that produces the most accurate pixel size.
                    malignancy['pixel_size_pct'] = malignancy.get('size_pct')
                    malignancy['pixel_region'] = malignancy.get('region')
                    if not mg_bbox_sam_ok:
                        malignancy['size_source'] = 'pixel'

                    # Location: trust the deterministic 3x3 grid label computed from
                    # the actual bbox (compute_anatomical_region). MedGemma's free-text
                    # tumor_location is unreliable (biased toward "right frontal lobe"
                    # regardless of where the tumor actually is) and is kept in
                    # malignancy.medgemma_assessment.tumor_location for transparency.
                    mg_loc = (mg.get('tumor_location') or '').strip()
                    if mg_loc:
                        prev_region = malignancy.get('region') or {}
                        # Only attach as a secondary annotation, do NOT overwrite label.
                        malignancy['region'] = {
                            **prev_region,
                            'medgemma_location': mg_loc,
                        }

                    # Size preference (rewritten 2026-05-28):
                    # The previous logic overrode the YOLO-measured pixel size
                    # with MedGemma's free-text gut estimate, which is unreliable
                    # — MedGemma routinely answers "1.5 %" regardless of actual
                    # tumor size, crushing the report for genuinely large
                    # lesions. New order of trust:
                    #   1. YOLO-bbox pixel size (compute_malignancy) — pixel%
                    #   2. MedGemma-bbox-SAM pixel size (independent measurement)
                    #   3. MedGemma's free-text gut estimate (only if both above
                    #      failed, e.g. YOLO returned 0 % and MG-bbox-SAM was
                    #      out of range)
                    pixel_size_pct = malignancy.get('pixel_size_pct')
                    mg_size_for_score = None
                    if pixel_size_pct is not None and pixel_size_pct > 0.5:
                        # Trust the YOLO-derived pixel size. Restore it as the
                        # canonical value (it may have been overwritten earlier
                        # by the MG-bbox-SAM step).
                        malignancy['size_pct'] = pixel_size_pct
                        malignancy['size_source'] = 'pixel'
                        mg_size_for_score = pixel_size_pct
                        malignancy['size_category'] = (
                            'small' if pixel_size_pct < 5.0
                            else ('medium' if pixel_size_pct < 20.0 else 'large')
                        )
                    elif mg_bbox_sam_ok and mg_bbox_sam_size_pct is not None:
                        malignancy['size_pct'] = mg_bbox_sam_size_pct
                        mg_size_for_score = mg_bbox_sam_size_pct
                        malignancy['size_category'] = (
                            'small' if mg_size_for_score < 5.0
                            else ('medium' if mg_size_for_score < 20.0 else 'large')
                        )
                    else:
                        try:
                            mg_size = float(mg.get('estimated_size_pct'))
                        except (TypeError, ValueError):
                            mg_size = None
                        if mg_size is not None and 0.0 < mg_size < 60.0:
                            malignancy['size_pct'] = round(mg_size, 1)
                            malignancy['size_source'] = 'medgemma'
                            mg_size_for_score = mg_size
                            cat = (mg.get('size_category') or '').strip().lower()
                            if cat in ('small', 'medium', 'large'):
                                malignancy['size_category'] = cat
                            else:
                                malignancy['size_category'] = (
                                    'small' if mg_size < 5.0
                                    else ('medium' if mg_size < 20.0 else 'large')
                                )
                    # Re-bucket the final size into a 5%-wide range for the UI.
                    _r = size_pct_to_range(malignancy.get('size_pct'))
                    if _r is not None:
                        malignancy['size_range']      = _r[2]
                        malignancy['size_range_low']  = _r[0]
                        malignancy['size_range_high'] = _r[1]
                    print(f'[SIZE-FINAL] size_pct={malignancy.get("size_pct")} '
                          f'range={malignancy.get("size_range")} '
                          f'source={malignancy.get("size_source")} '
                          f'bbox_src={malignancy.get("bbox_source")}', flush=True)
                    # Recompute the malignancy score with whichever size source won
                    # (MedGemma-bbox-SAM pixel size or MedGemma's free-text estimate).
                    if mg_size_for_score is not None and mg_size_for_score > 0:
                        base = BASE_RISK.get(class_name_for_recalc := uncertainty['class_name'],
                                             BASE_RISK['No Tumor'])
                        size_bonus = min((mg_size_for_score / 100.0) / 0.3, 1.0) * 2.0
                        raw = base['score'] + size_bonus
                        final = raw * (0.6 + 0.4 * float(uncertainty['mean_confidence']))
                        final = round(max(0.0, min(10.0, final)), 1)
                        malignancy['score'] = final
                        malignancy['score_out_of_10'] = f'{final}/10'
                        source_tag = 'MedGemma+SAM' if mg_bbox_sam_ok else 'MedGemma'
                        malignancy['summary'] = (
                            f"Type: {class_name_for_recalc} | Base risk: {base['label']} | "
                            f"~{mg_size_for_score:.1f}% ({source_tag}) | Malignancy: {final}/10"
                        )
            except Exception as mg_e:
                print(f'[MedGemma tumor] {mg_e}', flush=True)

        # Map model key -> display label + reported accuracy. Covers both
        # the v2 ensemble (preferred) and the v1 fallback.
        _DISPLAY = {
            'convnext_tiny':   ('ConvNeXt-Tiny',   '99.29%'),
            'efficientnet_b3': ('EfficientNet-B3', '99.48%'),
            'resnet50':        ('ResNet-50',       '99.34%'),
            # v1 legacy keys
            'densenet169':    ('DenseNet-169',    '98.80%'),
            'efficientnetb3': ('EfficientNet-B3', '99.10%'),
        }
        models_block = {}
        for k, res in model_results.items():
            label, acc = _DISPLAY.get(k, (k, '—'))
            models_block[k] = {'name': label, 'accuracy': acc, **res}

        # 3-way (or 2-way) agreement signal: how many models pick the same class
        from collections import Counter as _Counter
        preds_by_model = {k: v['class'] for k, v in model_results.items()}
        agree_counts = _Counter(preds_by_model.values())
        agree_top_class, agree_top_count = agree_counts.most_common(1)[0]
        agreement = {
            'predictions': preds_by_model,
            'total_models': len(preds_by_model),
            'agreeing_count': agree_top_count,
            'majority_class': agree_top_class,
            'unanimous': bool(agree_top_count == len(preds_by_model)),
        }
        # If not unanimous, that's an extra OOD / trust signal
        if not agreement['unanimous']:
            uncertainty['needs_review'] = True

        return jsonify({
            'success': True,
            'preprocessing': prep_info,
            'models': models_block,
            'agreement': agreement,
            'tta': {
                'enabled': True,
                'n_augmentations': len(TTA_OPS),
                'ops': TTA_OPS,
                'note': 'Per-model class probabilities are mean softmax across all augmentations. Energy OOD score uses the un-augmented forward pass.',
            },
            'best_model': best_name,
            'prediction': {
                # Use the TTA-averaged numbers as the headline so the prediction
                # card matches the Model Comparison chart. MC Dropout's posterior
                # mean still appears in the Uncertainty (Novel #1) panel.
                'class': model_results[best_name]['class'],
                'confidence': model_results[best_name]['confidence'],
                'probabilities': model_results[best_name]['probabilities'],
                'model_used': best_name,
                'source': 'tta_mean_softmax',
                # Secondary fields (MC Dropout) for transparency
                'mc_dropout_class': uncertainty['class_name'],
                'mc_dropout_confidence': uncertainty['mean_confidence'],
            },
            'uncertainty': uncertainty,
            'ood': tumor_ood_block,
            'xai': xai,

            'malignancy': malignancy,
            'anatomy_views': _generate_anatomy_views(original_img),
            'focus_crop': focus_crop_block,
            'robustness': robustness,
            'cross_dataset': {
                'brisc2025_test_acc': 98.80,
                'mendeley_test_acc': 95.20,
                'datasets_tested': 2,
                'validated': True
            },
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """Generate a diagnostic-impression via MedGemma (Ollama).

    Body: {
        "prediction": <full /api/predict response>,
        "language":   "en" | "es",
        "mode":       "basic" | "advanced"   # optional; omitted -> legacy single paragraph
    }
    """
    try:
        from llm.medgemma_client import generate_report
        payload = request.get_json(silent=True) or {}
        pred = payload.get('prediction')
        lang = (payload.get('language') or 'en').lower()
        mode_in = (payload.get('mode') or '').strip().lower() or None
        if not pred:
            return jsonify({'success': False, 'error': 'Missing "prediction" payload.'}), 400
        result = generate_report(pred, language=lang, mode=mode_in)
        status = 200 if result.get('success') else 502
        return jsonify(result), status
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Conversational chat with MedGemma. Patient/doctor audience, EN/ES.

    Body: {
        "messages":     [{"role": "user"|"assistant", "content": str}, ...],  # last ≤ 10
        "audience":     "patient" | "doctor",
        "language":     "en" | "es",
        "scan_context": { predicted_class, confidence, size_range, size_pct,
                          base_risk, score, location, side, symptoms[] } | null
    }
    """
    try:
        from llm.medgemma_client import chat as _chat
        payload = request.get_json(silent=True) or {}
        msgs = payload.get('messages') or []
        if not isinstance(msgs, list) or not msgs:
            return jsonify({'success': False, 'error': 'Missing "messages".'}), 400
        # Defensive caps: 10 turns, 4000 chars per message.
        msgs = msgs[-10:]
        for m in msgs:
            if isinstance(m.get('content'), str) and len(m['content']) > 4000:
                m['content'] = m['content'][:4000]
        audience = (payload.get('audience') or 'patient').lower()
        language = (payload.get('language') or 'en').lower()
        scan_ctx = payload.get('scan_context')
        result = _chat(messages=msgs, audience=audience, language=language,
                       scan_context=scan_ctx)
        status = 200 if result.get('success') else 502
        return jsonify(result), status
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/llm-health', methods=['GET'])
def llm_health():
    """Report whether Ollama is reachable and MedGemma is pulled."""
    try:
        from llm.medgemma_client import health
        return jsonify(health())
    except Exception as e:
        return jsonify({'ollama_reachable': False, 'error': str(e)}), 500


@app.route('/api/predict_manual_bbox', methods=['POST'])
def predict_manual_bbox():
    """Re-run classification + MedGemma assessment on a user-drawn region.

    Use case: the auto-detected tumor bbox (YOLO/CAM) is wrong. User draws a
    rectangle around the actual tumor. We crop the image to that rectangle
    (plus a little context padding), run all 3 classifiers on the crop, and
    send the crop to MedGemma for a textual assessment.

    Request: multipart form with
        image  — original uploaded image file
        bbox   — JSON string: {"x": int, "y": int, "w": int, "h": int}
                 coordinates are in the ORIGINAL image's pixel space.
    Response: subset of /api/predict shaped for swap-in:
        prediction.{class, confidence, probabilities}
        agreement.{unanimous, agreeing_count, total_models}
        models.{key.{class, confidence, probabilities, name}}
        uncertainty.{class_name, mean_confidence, epistemic, ci_lower, ci_upper, probabilities}
        malignancy.{score, base_risk, size_pct (=NA), bbox, tumor_crop_image, medgemma_assessment}
    """
    try:
        import json as _json
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        if 'bbox' not in request.form:
            return jsonify({'error': 'No bbox provided'}), 400
        lang = request.form.get('language', 'en')
        try:
            bbox_in = _json.loads(request.form['bbox'])
            bx = int(bbox_in['x']); by = int(bbox_in['y'])
            bw = int(bbox_in['w']); bh = int(bbox_in['h'])
        except Exception:
            return jsonify({'error': 'Malformed bbox JSON'}), 400
        if bw < 8 or bh < 8:
            return jsonify({'error': 'Drawn box is too small (min 8x8 px)'}), 400

        image_bytes = request.files['image'].read()
        if not image_bytes:
            return jsonify({'error': 'Empty image file'}), 400

        original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        W, H = original.size

        # Crop the user's bbox + 10% context padding
        pad_x = max(4, int(bw * 0.10))
        pad_y = max(4, int(bh * 0.10))
        left   = max(0, bx - pad_x)
        top    = max(0, by - pad_y)
        right  = min(W, bx + bw + pad_x)
        bottom = min(H, by + bh + pad_y)
        crop = original.crop((left, top, right, bottom))

        # Preprocess the crop as if it were the input image
        crop_buf = io.BytesIO()
        crop.save(crop_buf, format='PNG')
        crop_bytes = crop_buf.getvalue()
        image_tensor, _crop_pil, _prep_info = preprocess_image(crop_bytes)
        all_models, device = get_models()
        image_tensor = image_tensor.to(device)

        # Run all 3 classifiers (single pass each — no TTA, this is a quick op)
        model_results = {}
        KEY_TO_NAME = {
            'densenet169': 'DenseNet-169',
            'efficientnetb3': 'EfficientNet-B3',
            'resnet50': 'ResNet-50',
        }
        for mname, model in all_models.items():
            model.eval()
            with torch.no_grad():
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())
            model_results[mname] = {
                'name': KEY_TO_NAME.get(mname, mname),
                'class': CLASS_NAMES[pred_class],
                'confidence': round(float(probs[pred_class]), 6),
                'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(4)},
                'pred_index': pred_class,
                'tta_applied': False,
            }

        # Agreement
        pred_classes = [r['class'] for r in model_results.values()]
        unanimous = len(set(pred_classes)) == 1
        from collections import Counter
        cnt = Counter(pred_classes)
        majority_class, agreeing = cnt.most_common(1)[0]

        # Best model = highest confidence on its predicted class
        best_name = max(model_results, key=lambda k: model_results[k]['confidence'])
        best_model = all_models[best_name]
        uncertainty = predict_with_uncertainty(best_model, image_tensor, device, T=20)

        # MedGemma on the crop. bbox_norm in 0..1 of the crop = full crop here.
        medgemma_assessment = None
        medgemma_model_name = None
        try:
            if uncertainty['class_name'] != 'No Tumor':
                from llm.medgemma_client import assess_tumor as _assess_tumor
                _b64 = base64.b64encode(crop_bytes).decode('utf-8')
                mg = _assess_tumor(
                    image_b64=_b64,
                    class_name=uncertainty['class_name'],
                    confidence=uncertainty['mean_confidence'],
                    bbox_224={'x': 0, 'y': 0, 'w': 224, 'h': 224},
                    size_pct=0.0,
                    language=lang,
                )
                if mg.get('success') and mg.get('assessment'):
                    medgemma_assessment = mg['assessment']
                    medgemma_model_name = mg.get('model')
        except Exception as _mg_e:
            print(f'[manual-bbox] medgemma assessment failed: {_mg_e}', flush=True)

        # Build the display crop image (no extra bbox drawn — the user already saw their box)
        # Make it a reasonable size: longest side ~360 px.
        crop_disp = crop.copy()
        cw, ch = crop_disp.size
        max_side = 360
        if max(cw, ch) > max_side:
            scale = max_side / max(cw, ch)
            crop_disp = crop_disp.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
        _disp_buf = io.BytesIO(); crop_disp.save(_disp_buf, format='PNG')
        crop_b64 = base64.b64encode(_disp_buf.getvalue()).decode('utf-8')

        # Risk baseline for malignancy summary
        cls_name = uncertainty['class_name']
        base = get_base_risk(cls_name, lang)
        # Without YOLO/SAM we don't have a pixel-tight mask, so size_pct is unknown.
        # Score from base risk + confidence (no size term).
        final_score = round(max(0.0, min(10.0, base['score'] * (0.6 + 0.4 * float(uncertainty['mean_confidence'])))), 1)

        # Anatomical region: map the user-drawn bbox (in original-image px) into
        # the 224x224 frame the grid uses, then look up the deterministic label.
        bbox_in_224 = {
            'x': int(round(bx * 224 / W)),
            'y': int(round(by * 224 / H)),
            'w': max(1, int(round(bw * 224 / W))),
            'h': max(1, int(round(bh * 224 / H))),
        }
        anatomical = compute_anatomical_region(bbox_in_224, cls_name, lang=lang)

        return jsonify({
            'source': 'manual_bbox',
            'manual_bbox_original': {'x': bx, 'y': by, 'w': bw, 'h': bh},
            'prediction': {
                'class': uncertainty['class_name'],
                'confidence': uncertainty['mean_confidence'],
                'probabilities': uncertainty['probabilities'],
            },
            'best_model': best_name,
            'models': model_results,
            'agreement': {
                'unanimous': bool(unanimous),
                'agreeing_count': int(agreeing),
                'total_models': len(model_results),
                'majority_class': majority_class,
            },
            'uncertainty': uncertainty,
            'malignancy': {
                'score': final_score,
                'score_out_of_10': f'{final_score}/10',
                'base_risk': base['label'],
                'base_score': base['score'],
                'clinical_note': base['note'],
                'size_pct': None,
                'size_category': 'unknown',
                'size_method': 'Manual region (user-drawn) — size %% not computed',
                'bbox': bbox_in_224,                 # in 224x224 frame for downstream consumers
                'bbox_source': 'manual',
                'region': anatomical,                # grid-based anatomical label
                'clinical_context': (lambda _ctx: ({
                    'primer':             _ctx['primer'],
                    'size_interpretation': None,    # unknown for manual region
                    'subtypes':           _ctx['subtypes'],
                    'typical_workup':     _ctx['typical_workup'],
                }) if _ctx else None)(get_clinical_context(cls_name, lang)),
                'urgency':       compute_urgency(cls_name, final_score, None, lang=lang),
                'medical_codes': get_medical_codes(cls_name, lang=lang),
                'tumor_crop_image': crop_b64,
                'tumor_crop_size': list(crop_disp.size),
                'medgemma_assessment': medgemma_assessment,
                'medgemma_model': medgemma_model_name,
                'convention': 'radiological (patient left = image right)',
                'summary': f"Type: {cls_name} | Base risk: {base['label']} | Manual region | Malignancy: {final_score}/10",
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


_metrics_cache = None
_METRICS_CACHE_FILE = Path(__file__).parent / '.metrics_cache.json'


def _metrics_cache_fingerprint() -> str:
    """Build a fingerprint of inputs that affect the cached metrics.

    If any model .pth file mtime or the GPU name changes, the cache is stale.
    Cheap to compute on every request (just stat() calls).
    """
    parts = []
    models_dir = Path(__file__).parent / 'models'
    v2_dir = models_dir / 'v2'
    if v2_dir.exists():
        for name in sorted(['convnext_tiny.pth', 'efficientnet_b3.pth', 'resnet50.pth']):
            p = v2_dir / name
            parts.append(f'v2/{name}:{int(p.stat().st_mtime) if p.exists() else 0}')
        parts.append(f'gpu:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
        return '|'.join(parts)
    for name in sorted(['densenet169_best.pth', 'efficientnetb3_best.pth', 'resnet50_best.pth']):
        p = models_dir / name
        parts.append(f'{name}:{int(p.stat().st_mtime) if p.exists() else 0}')
    parts.append(f'gpu:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    return '|'.join(parts)


def _load_metrics_cache_from_disk() -> None:
    """If a prior run saved metrics to disk and inputs haven't changed, restore them."""
    global _metrics_cache
    if not _METRICS_CACHE_FILE.exists():
        return
    try:
        import json as _json
        with _METRICS_CACHE_FILE.open('r', encoding='utf-8') as f:
            blob = _json.load(f)
        if blob.get('fingerprint') == _metrics_cache_fingerprint():
            _metrics_cache = blob.get('payload')
            print(f'[metrics] restored disk cache ({_METRICS_CACHE_FILE.name})', flush=True)
    except Exception as e:
        print(f'[metrics] could not read disk cache: {e}', flush=True)


def _save_metrics_cache_to_disk(payload: dict) -> None:
    try:
        import json as _json
        with _METRICS_CACHE_FILE.open('w', encoding='utf-8') as f:
            _json.dump({'fingerprint': _metrics_cache_fingerprint(), 'payload': payload}, f)
        print(f'[metrics] saved disk cache', flush=True)
    except Exception as e:
        print(f'[metrics] could not save disk cache: {e}', flush=True)


def _parse_metrics_file(path: Path) -> dict | None:
    """Parse a *_test_metrics.txt file produced by src/train_combined.py.

    Also reads a sibling *_full_metrics.json file (from src/evaluate_full_metrics.py)
    when present, adding precision / recall / F1 / confusion matrix.
    """
    if not path.exists():
        return None
    out: dict = {'per_class': {}}
    try:
        for raw in path.read_text(encoding='utf-8').splitlines():
            s = raw.strip()
            if not s:
                continue
            if 'val balanced_acc' in s:
                out['val_balanced_acc'] = float(s.split(':')[-1].strip())
            elif s.startswith('test  loss'):
                out['test_loss'] = float(s.split(':')[-1].strip())
            elif s.startswith('test  accuracy'):
                out['test_accuracy'] = float(s.split(':')[-1].strip())
            elif s.startswith('test  balanced_acc'):
                out['test_balanced_acc'] = float(s.split(':')[-1].strip())
            elif s.startswith(('Glioma', 'Meningioma', 'No Tumor', 'Pituitary')) and 'acc=' in s:
                # e.g. "Glioma       acc=0.9888" — this is per-class RECALL, not accuracy.
                parts = s.split('acc=')
                cls = parts[0].strip()
                acc = float(parts[1].strip())
                out['per_class'][cls] = {'recall': acc}

        # Merge full-metrics JSON if present (precision / F1 / confusion matrix)
        full_path = path.with_name(path.stem.replace('_test_metrics', '_full_metrics') + '.json')
        if full_path.exists():
            try:
                import json as _json
                full = _json.loads(full_path.read_text(encoding='utf-8'))
                for cname, stats in (full.get('per_class') or {}).items():
                    existing = out['per_class'].get(cname) or {}
                    out['per_class'][cname] = {**existing, **stats}
                for k in ('macro_precision', 'macro_recall', 'macro_f1',
                          'weighted_precision', 'weighted_recall', 'weighted_f1',
                          'confusion_matrix', 'class_order'):
                    if k in full:
                        out[k] = full[k]
            except Exception as _e:
                print(f'[metrics] could not merge {full_path.name}: {_e}', flush=True)

        return out if 'test_accuracy' in out else None
    except Exception:
        return None


def _measure_latency(model, device, n_warmup: int = 1, n_iters: int = 3) -> dict:
    """Time forward passes on a 1x3x224x224 dummy tensor.

    Defaults dropped from (warmup=2, iters=8) -> (1, 3) — std on a calibrated
    GPU is ~1-2 ms anyway, three samples give a stable enough number for the
    Benchmarks card and the loop runs ~3x faster.
    """
    import time as _time
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        times_single = []
        for _ in range(n_iters):
            t0 = _time.perf_counter()
            _ = model(x)
            times_single.append((_time.perf_counter() - t0) * 1000.0)
        # TTA = 5 passes per inference (matches what /api/predict does)
        times_tta = []
        for _ in range(n_iters):
            t0 = _time.perf_counter()
            for _ in range(5):
                _ = model(x)
            times_tta.append((_time.perf_counter() - t0) * 1000.0)
    arr_s = np.array(times_single)
    arr_t = np.array(times_tta)
    return {
        'single_pass_ms_mean':  float(arr_s.mean()),
        'single_pass_ms_std':   float(arr_s.std()),
        'tta5_ms_mean':         float(arr_t.mean()),
        'tta5_ms_std':          float(arr_t.std()),
        'n_iters':              n_iters,
    }


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


_V2_METRICS_FILE = Path(__file__).parent.parent / 'reports' / 'v2_metrics.json'

# Display labels for the v2 models — used by /api/metrics when v2_metrics.json
# is the data source. Order here is the order shown in the UI.
_V2_DISPLAY = [
    ('convnext_tiny',   'ConvNeXt-Tiny'),
    ('efficientnet_b3', 'EfficientNet-B3'),
    ('resnet50',        'ResNet-50'),
]


def _build_v2_metrics_payload(device) -> dict | None:
    """If reports/v2_metrics.json exists, build the /api/metrics response from
    it. This is the authoritative source after the v2 training; it carries
    test accuracy, balanced accuracy, per-class P/R/F1, ECE, latency, params,
    confusion matrices, AND ensemble agreement stats."""
    if not _V2_METRICS_FILE.exists():
        return None
    try:
        import json as _json
        blob = _json.loads(_V2_METRICS_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[metrics] could not parse v2_metrics.json: {e}', flush=True)
        return None

    # Backend class keys -> UI display labels
    CLS_LABEL = {
        'glioma':     'Glioma',
        'meningioma': 'Meningioma',
        'notumor':    'No Tumor',
        'pituitary':  'Pituitary',
    }
    entries = []
    by_name = {m['name']: m for m in (blob.get('models') or [])}
    for key, display in _V2_DISPLAY:
        m = by_name.get(key)
        if not m:
            continue
        # Transform per_class array -> dict keyed by display label so the
        # existing recall chart and any per-class table pick it up directly.
        per_class_arr = m.get('per_class') or []
        per_class_dict = {
            CLS_LABEL.get(row.get('class'), row.get('class')): {
                'precision': row.get('precision'),
                'recall':    row.get('recall'),
                'f1':        row.get('f1'),
                'support':   row.get('support'),
            }
            for row in per_class_arr
        }
        entries.append({
            'key':            key,
            'name':           display,
            'arch':           key.replace('_', '-'),
            'size_mb':        m.get('model_size_mb'),
            'parameters':     int((m.get('param_count_M') or 0) * 1e6),
            'parameters_M':   m.get('param_count_M'),
            'latency': {
                'single_pass_ms_mean': m.get('latency_ms_single'),
                'single_pass_ms_std':  0.0,
                'tta5_ms_mean':        m.get('latency_ms_tta5'),
                'tta5_ms_std':         0.0,
            },
            'test_metrics': {
                'test_accuracy':       m.get('test_accuracy'),
                'test_balanced_acc':   m.get('test_balanced_accuracy'),
                'ece':                 m.get('ece'),
                'per_class':           per_class_dict,
                'confusion_matrix':    m.get('confusion_matrix') or [],
                'confusion_png':       m.get('confusion_png'),
            },
        })
    return {
        'device':    str(device),
        'cuda_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'source':    'v2',
        'classes':   blob.get('classes', []),
        'ensemble':  blob.get('ensemble', {}),
        'models':    entries,
        'note':      ('v2 metrics — locked test set (2 114 images). Latency measured on '
                      '1x3x224x224 with 10 warmup + 30 timed passes. ECE computed with '
                      '15 confidence bins. Ensemble = soft-vote (mean softmax across the '
                      'three models). Pairwise / 3-way agreement = top-1 prediction match.'),
    }


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Return a comparison table of all loaded tumor models.

    Prefers reports/v2_metrics.json (produced by src/eval_v2.py) when present
    — that gives test accuracy, balanced accuracy, per-class P/R/F1, ECE,
    confusion matrices, and ensemble agreement.

    Falls back to the legacy combined-training reports + live latency
    measurement when v2_metrics.json isn't there yet.
    """
    global _metrics_cache
    if _metrics_cache is not None and request.args.get('refresh') != '1':
        return jsonify(_metrics_cache)

    all_models, device = get_models()

    # ── v2 path: take everything from the single JSON ─────────────
    v2_payload = _build_v2_metrics_payload(device)
    if v2_payload is not None:
        _metrics_cache = v2_payload
        _save_metrics_cache_to_disk(v2_payload)
        return jsonify(v2_payload)

    # ── Legacy path: per-model live measurement + txt parsing ─────
    reports_dir = Path(__file__).parent.parent / 'reports' / 'combined'
    models_dir  = Path(__file__).parent / 'models'

    KEY_TO_FILE = {
        'densenet169':    ('densenet169_best.pth',    'DenseNet-169'),
        'efficientnetb3': ('efficientnetb3_best.pth', 'EfficientNet-B3'),
        'resnet50':       ('resnet50_best.pth',       'ResNet-50'),
    }
    entries = []
    for key, model in all_models.items():
        filename, display = KEY_TO_FILE.get(key, (None, key))
        size_mb = None
        if filename:
            p = models_dir / filename
            if p.exists():
                size_mb = round(p.stat().st_size / (1024 * 1024), 2)
        test_metrics = _parse_metrics_file(reports_dir / f'{key}_test_metrics.txt')
        try:
            latency = _measure_latency(model, device)
        except Exception as e:
            latency = {'error': str(e)}
        entries.append({
            'key':            key,
            'name':           display,
            'arch':           model.__class__.__name__,
            'size_mb':        size_mb,
            'parameters':     _count_params(model),
            'parameters_M':   round(_count_params(model) / 1e6, 2),
            'latency':        latency,
            'test_metrics':   test_metrics,
        })

    payload = {
        'device':       str(device),
        'cuda_name':    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'source':       'legacy',
        'models':       entries,
        'note':         ('Legacy metrics — test results read from '
                         'reports/combined/{model}_test_metrics.txt. Run src/eval_v2.py '
                         'to upgrade to the v2 metrics shown after retraining.'),
    }
    _metrics_cache = payload
    _save_metrics_cache_to_disk(payload)
    return jsonify(payload)


@app.route('/api/reports/<path:filename>', methods=['GET'])
def reports_static(filename):
    """Serve files from the project-root `reports/` directory.

    Used for confusion-matrix and calibration figures referenced from
    /api/metrics. Path-traversal guarded — anything resolving outside
    `reports/` returns 404.
    """
    base = (Path(__file__).parent.parent / 'reports').resolve()
    target = (base / filename).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return jsonify({'error': 'forbidden'}), 404
    if not target.exists() or not target.is_file():
        return jsonify({'error': 'not found'}), 404
    return send_file(str(target))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


# ── Sample Images API ──────────────────────────────────────────────
DATASETS_META = {
    'kaggle_mri': {
        'name': 'Kaggle Brain Tumor MRI Dataset',
        'source': 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset',
        'description': 'Brain Tumor MRI Dataset, a widely used benchmark for brain tumor classification. After MD5 de-duplication against the other two sources, 1,145 unique images from this dataset are used in the combined training/validation/test split.',
        'total_images': 1145,
        'badge_color': '#667eea',
    },
    'brisc2025': {
        'name': 'BRISC-2025 Challenge Dataset',
        'source': 'https://www.kaggle.com/datasets/briscdataset/brisc2025',
        'description': 'Brain tumor Image Segmentation and Classification 2025 challenge dataset. Our primary training dataset with high-quality annotated MRI scans. 5,950 unique images from this dataset are used after de-duplication.',
        'total_images': 5950,
        'badge_color': '#38ef7d',
    },
    'hospital': {
        'name': 'Mendeley Hospital MRI Dataset',
        'source': 'https://data.mendeley.com/datasets/zwr4ntf94j/5',
        'description': 'Epic and CSCR Hospital Dataset of clinical MRI scans. Used for cross-dataset generalization validation (zero-retraining). 7,000 unique images from this dataset are used after de-duplication.',
        'total_images': 7000,
        'badge_color': '#f5a623',
    },
}

TUMOR_INFO = {
    'glioma': {'label': 'Glioma', 'description': 'Arises from glial cells. Most common and aggressive primary brain tumor. Subtypes include astrocytoma, oligodendroglioma.'},
    'meningioma': {'label': 'Meningioma', 'description': 'Grows from meninges (brain/spinal cord membranes). Usually benign and slow-growing. Most common benign brain tumor.'},
    'pituitary': {'label': 'Pituitary Tumor', 'description': 'Develops in the pituitary gland at the base of the brain. Can affect hormone production. Usually treatable.'},
    'no_tumor': {'label': 'No Tumor (Healthy)', 'description': 'Normal brain MRI scan with no detectable tumor. Used as baseline for classification model training.'},
}


@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Return list of all sample images grouped by dataset and class."""
    samples_dir = os.path.join(app.static_folder, 'samples')
    result = {}
    for ds_key, ds_meta in DATASETS_META.items():
        ds_path = os.path.join(samples_dir, ds_key)
        if not os.path.isdir(ds_path):
            continue
        classes = {}
        for cls in ['glioma', 'meningioma', 'pituitary', 'no_tumor']:
            cls_path = os.path.join(ds_path, cls)
            if not os.path.isdir(cls_path):
                continue
            files = sorted([f for f in os.listdir(cls_path) if not f.startswith('.')])
            classes[cls] = {
                'info': TUMOR_INFO.get(cls, {}),
                'images': [f'/samples/{ds_key}/{cls}/{f}' for f in files],
            }
        result[ds_key] = {**ds_meta, 'classes': classes}
    return jsonify(result)


@app.route('/api/samples/download/<dataset>/<tumor_class>/<filename>', methods=['GET'])
def download_sample(dataset, tumor_class, filename):
    """Serve a sample image for download / use as scan input."""
    safe_ds = os.path.basename(dataset)
    safe_cls = os.path.basename(tumor_class)
    safe_fn = os.path.basename(filename)
    samples_dir = os.path.join(app.static_folder, 'samples', safe_ds, safe_cls)
    return send_from_directory(samples_dir, safe_fn)


# ── Serve React Frontend ───────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


def _prewarm_metrics_cache() -> None:
    """Run /api/metrics' compute path in-process to fill _metrics_cache.

    Saves the first user click on Benchmarks from waiting for latency
    measurements (which take a few seconds even at the reduced loop size).
    Runs in a background thread so it doesn't delay Flask boot.
    """
    global _metrics_cache
    if _metrics_cache is not None:
        return  # disk cache already restored it
    try:
        with app.test_request_context('/api/metrics'):
            metrics()  # populates _metrics_cache + writes to disk
        print('[metrics] pre-warmed at boot', flush=True)
    except Exception as e:
        print(f'[metrics] pre-warm failed: {e}', flush=True)


if __name__ == '__main__':
    import sys
    import threading
    print("Pre-loading models...", flush=True)
    try:
        get_models()
    except Exception as e:
        print(f"ERROR loading models: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    # Restore prior metrics cache from disk if model weights are unchanged
    _load_metrics_cache_from_disk()
    # If no valid disk cache, kick off a background pre-warm so the first
    # /api/metrics request finds a ready payload
    if _metrics_cache is None:
        threading.Thread(target=_prewarm_metrics_cache, daemon=True).start()
    print("Starting server on port 7860...", flush=True)
    app.run(host='0.0.0.0', port=7860)
