"""
xai.py
======
Section 6 (interpretability half): Grad-CAM for the CNN branch and an
attention-rollout style heatmap for the Swin branch, fused into a single
overlay so the clinician-facing output highlights both fine local lesions
(from EfficientNet-B3) and global structural context (from Swin), e.g.
optic-disc boundary emphasis for Glaucoma or vascular/exudate emphasis
for DR.
"""

import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Standard Grad-CAM hooked onto the last convolutional layer of the
    CNN branch (EfficientNet-B3 features).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # Global-average-pool the gradients -> per-channel importance weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)            # (B, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze(0).cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _swin_attention_map(model: torch.nn.Module) -> np.ndarray:
    """
    Builds a coarse spatial saliency map from the Swin branch's cached
    last-stage token map (channel-wise L2 norm per spatial location acts
    as a simple, hook-free proxy for "how much the global-context branch
    reacted at this location" -- cheap and robust across torchvision Swin
    versions that don't uniformly expose raw attention weights).
    """
    feats = model._last_swin_feats  # (B, C, H, W), cached in RetinalSwinNet.forward
    if feats is None:
        raise RuntimeError("Run a forward pass through the model before requesting the Swin map.")
    saliency = feats.norm(dim=1).squeeze(0).detach().cpu().numpy()  # (H, W)
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


def _overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resizes a [0,1] heatmap to the image size and alpha-blends a JET colormap over it."""
    w, h = pil_image.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    base = np.array(pil_image.convert("RGB")).astype(np.float32)
    overlay = (alpha * heatmap_color.astype(np.float32) + (1 - alpha) * base).astype(np.uint8)
    return overlay


def generate_attention_heatmap(
    model: torch.nn.Module,
    image_path: str,
    transform,
    class_names,
    device: str = "cpu",
    save_path: Optional[str] = None,
):
    """
    Full pipeline: load a fundus image -> preprocess -> forward pass ->
    Grad-CAM (CNN branch) + Swin token-norm saliency (Transformer branch)
    -> fused overlay highlighting the regions driving the prediction.

    Args:
        model: a trained RetinalSwinNet.
        image_path: path to a single fundus image on disk.
        transform: an eval-mode RetinalPreprocessTransform (no augmentation).
        class_names: list of class name strings, indexed by class id.
        device: 'cpu' or 'cuda'.
        save_path: if provided, writes the overlay PNG to this path.

    Returns:
        dict with keys: 'predicted_class', 'confidence', 'overlay' (ndarray).
    """
    model.eval()
    model.to(device)

    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Target the last conv block of the EfficientNet-B3 branch for Grad-CAM.
    target_layer = model.cnn_branch.features[-1]
    cam_extractor = GradCAM(model, target_layer)

    with torch.enable_grad():
        cam, pred_class = cam_extractor(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    swin_map = _swin_attention_map(model)
    cam_extractor.remove_hooks()

    # Resize both maps to a common grid, then average (equal-weight fusion
    # of local lesion cues and global context).
    common_size = (14, 14)
    cam_resized = cv2.resize(cam, common_size)
    swin_resized = cv2.resize(swin_map, common_size)
    fused_map = 0.5 * cam_resized + 0.5 * swin_resized

    overlay = _overlay_heatmap(pil_image, fused_map)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "predicted_class": class_names[pred_class],
        "confidence": float(probs[pred_class]),
        "class_probabilities": {name: float(p) for name, p in zip(class_names, probs)},
        "overlay": overlay,
    }
