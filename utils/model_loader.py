"""
AGWFN Ensemble Model Loader
----------------------------
Exact reconstruction of the AttentionFusionEnsemble class used during training
(see Untitled2.ipynb, Cell "CELL 6 — Attention Fusion Ensemble").

4 frozen backbones (ViT-Small, EfficientNet-B2, ResNet50, MobileNetV2)
feed into a learned attention-guided fusion head:
    - global_w      (4,)     global per-model weight
    - class_attn    (4, 6)   per-class attention across models
    - temperature   (6,)     per-class temperature scaling
    - head          MLP      256->128->64->6 calibration head (LayerNorm+GELU)
    - blend         scalar   residual blend between raw fused logits and head output

Checkpoint format (ensemble_best.pth):
    {
        "model_state": OrderedDict[...],   # full AttentionFusionEnsemble state_dict
        "val_loss": float,
        "val_f1": float,
        "weights": {...},                  # informational snapshot from training, not used for loading
        "epoch": int,
    }
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as tvm
import timm

# Order matches training exactly: CLASSES = ["book","cell phone","headphone","laptop","person","tv"]
CLASS_NAMES = ["Book", "Cell Phone", "Headphone", "Laptop", "Person", "Television"]
IMG_SIZE = 224
DECISION_THRESHOLD = 0.45


def build_vit_small(num_classes: int = len(CLASS_NAMES)):
    return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)


def build_efficientnet(num_classes: int = len(CLASS_NAMES)):
    return timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes, drop_rate=0.4)


def build_resnet50(num_classes: int = len(CLASS_NAMES)):
    m = tvm.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
    return m


def build_mobilenetv2(num_classes: int = len(CLASS_NAMES)):
    m = tvm.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, num_classes))
    return m


class AttentionFusionEnsemble(nn.Module):
    """Exact match to the training-time class. Attribute names must match the
    checkpoint's state_dict keys (m_vit, m_eff, m_res, m_mob, global_w,
    class_attn, temperature, head, blend) — do not rename."""

    def __init__(self, m_vit, m_eff, m_res, m_mob, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        for m in [m_vit, m_eff, m_res, m_mob]:
            for p in m.parameters():
                p.requires_grad = False

        self.m_vit = m_vit
        self.m_eff = m_eff
        self.m_res = m_res
        self.m_mob = m_mob
        self.num_classes = num_classes

        self.global_w = nn.Parameter(torch.ones(4) / 4.0)
        self.class_attn = nn.Parameter(torch.ones(4, num_classes) / 4.0)
        self.temperature = nn.Parameter(torch.ones(num_classes))

        self.head = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

        self.blend = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        with torch.no_grad():
            l_vit = self.m_vit(x)
            l_eff = self.m_eff(x)
            l_res = self.m_res(x)
            l_mob = self.m_mob(x)

        stack = torch.stack([l_vit, l_eff, l_res, l_mob], dim=1)  # (B, 4, 6)

        g = torch.softmax(self.global_w, dim=0)         # (4,)
        c = torch.softmax(self.class_attn, dim=0)        # (4, 6)
        w = g.unsqueeze(1) * c
        w = w / (w.sum(dim=0, keepdim=True) + 1e-8)       # (4, 6)

        raw = (stack * w.unsqueeze(0)).sum(dim=1)         # (B, 6) fused logits

        temp = torch.clamp(self.temperature, min=0.5, max=2.0)
        raw_scaled = raw / temp.unsqueeze(0)

        calibrated = self.head(raw_scaled)

        b = torch.sigmoid(self.blend)
        out = b * raw_scaled + (1.0 - b) * calibrated
        return out  # logits — apply sigmoid for probabilities

    def get_weights(self):
        g = torch.softmax(self.global_w, dim=0).detach().cpu().numpy()
        return {
            "ViT-Small": round(float(g[0]), 4),
            "EfficientNet": round(float(g[1]), 4),
            "ResNet50": round(float(g[2]), 4),
            "MobileNetV2": round(float(g[3]), 4),
        }


def load_checkpoint(path: str, device: str = "cpu") -> AttentionFusionEnsemble:
    """Loads ensemble_best.pth into a ready-to-use AttentionFusionEnsemble (eval mode)."""
    device = torch.device(device)

    m_vit = build_vit_small()
    m_eff = build_efficientnet()
    m_res = build_resnet50()
    m_mob = build_mobilenetv2()

    model = AttentionFusionEnsemble(m_vit, m_eff, m_res, m_mob)

    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[AGWFN loader] loaded with mismatches — "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval().to(device)
    return model


@torch.no_grad()
def run_inference(model: AttentionFusionEnsemble, tensor: torch.Tensor) -> np.ndarray:
    """tensor: (1, 3, 224, 224). Returns fused sigmoid probabilities, shape (6,)."""
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs


def decide_alert(probs: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Paper's decision rule:
      - MALPRACTICE if any prohibited object class fires
      - NORMAL if only 'Person' fires
      - WARNING-NO CANDIDATE VISIBLE if 'Person' does not fire
    """
    flags = {cls: bool(p >= threshold) for cls, p in zip(CLASS_NAMES, probs)}
    prohibited = ["Cell Phone", "Book", "Laptop", "Headphone", "Television"]

    if not flags["Person"]:
        state = "WARNING-NO CANDIDATE VISIBLE"
    elif any(flags[c] for c in prohibited):
        state = "MALPRACTICE"
    else:
        state = "NORMAL"

    return {"state": state, "flags": flags, "probs": dict(zip(CLASS_NAMES, probs.tolist()))}
