"""Preprocessing helpers to turn a raw webcam frame (numpy BGR/RGB array)
into a normalized tensor batch for AGWFN inference."""

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from .model_loader import IMG_SIZE

# Standard ImageNet normalization — matches typical ViT/EfficientNet/ResNet/MobileNet training
_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """frame: HxWx3 RGB uint8 numpy array (as given by streamlit-webrtc / camera_input)."""
    img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    tensor = _TRANSFORM(img)
    return tensor.unsqueeze(0)  # add batch dim
