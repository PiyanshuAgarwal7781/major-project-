"""
preprocessing.py
=================
Section 2: Custom Preprocessing & Augmentation Pipeline.

Implements:
  1. Circular fundus ROI auto-cropping (removes the black background
     surrounding the circular retinal photograph).
  2. Green-channel extraction + Adaptive CLAHE, re-projected back into a
     3-channel image so it stays compatible with ImageNet-pretrained
     backbones.
  3. Clinical-grade augmentation (flips, affine rotation, mild color jitter).
  4. ImageNet normalization.

`RetinalPreprocessTransform` is a single callable usable as a torchvision
transform: it can be dropped straight into `transforms.Compose([...])` or
used stand-alone in a `Dataset.__getitem__`.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


# --------------------------------------------------------------------- #
# Step 1: Circular fundus ROI auto-crop
# --------------------------------------------------------------------- #
def crop_fundus_roi(image_bgr: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop tightly around the circular fundus region, discarding the black
    background that fundus cameras typically produce around the retina.

    Args:
        image_bgr: HxWx3 uint8 image in BGR order (OpenCV convention).
        threshold: pixel intensity below which a pixel is considered
                   "background" in the grayscale mask.

    Returns:
        Cropped HxWx3 uint8 BGR image. If no meaningful foreground is
        found (e.g. a near-black image), the original image is returned
        unchanged to avoid crashing the pipeline.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Clean small noise specks so the bounding box isn't thrown off by
    # stray bright pixels outside the actual fundus circle.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_bgr

    x, y, w, h = cv2.boundingRect(coords)
    if w < 10 or h < 10:  # degenerate crop guard
        return image_bgr

    return image_bgr[y:y + h, x:x + w]


# --------------------------------------------------------------------- #
# Step 2: Green-channel extraction + Adaptive CLAHE
# --------------------------------------------------------------------- #
def apply_green_channel_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Extract the green channel (highest vascular / lesion contrast in
    fundus photography), apply CLAHE, then re-merge the enhanced channel
    back across all three RGB channels so downstream ImageNet-pretrained
    backbones (which expect 3-channel input) still function correctly.

    Args:
        image_bgr: HxWx3 uint8 BGR image.
        clip_limit: CLAHE contrast clipping threshold.
        tile_grid_size: CLAHE local neighborhood tile size.

    Returns:
        HxWx3 uint8 BGR image, contrast-enhanced via the green channel.
    """
    # OpenCV order is BGR -> green channel is index 1.
    green = image_bgr[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    green_eq = clahe.apply(green)

    # Re-project the enhanced green channel onto all 3 channels. This
    # preserves the "3-channel RGB-like" shape backbones expect, while
    # letting the highest-contrast structural information (vessels, optic
    # disc rim, hemorrhages) dominate what the network sees.
    enhanced = cv2.merge([green_eq, green_eq, green_eq])
    return enhanced


# --------------------------------------------------------------------- #
# Combined callable transform
# --------------------------------------------------------------------- #
class RetinalPreprocessTransform:
    """
    End-to-end preprocessing + augmentation transform for retinal fundus
    images. Accepts a PIL.Image (RGB) and returns a normalized torch
    tensor ready for the model.

    Pipeline:
        PIL (RGB) -> BGR ndarray -> ROI crop -> green-channel CLAHE
                   -> PIL -> resize -> [augmentations if train=True]
                   -> ToTensor -> Normalize(ImageNet stats)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_size: int = 300,
        train: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.image_size = image_size
        self.train = train
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        # Geometric + color augmentations, applied only during training.
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])

        self.to_tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        # 1. PIL RGB -> OpenCV BGR ndarray
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 2. Circular ROI crop
        bgr = crop_fundus_roi(bgr)

        # 3. Green-channel CLAHE enhancement
        bgr = apply_green_channel_clahe(
            bgr,
            clip_limit=self.clahe_clip_limit,
            tile_grid_size=self.clahe_tile_grid_size,
        )

        # 4. Back to PIL RGB for torchvision transforms
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(rgb)
        pil_out = TF.resize(pil_out, [self.image_size, self.image_size])

        # 5. Augmentation (train only)
        if self.train:
            pil_out = self.augment(pil_out)

        # 6. Tensor + normalize
        return self.to_tensor_normalize(pil_out)


def build_transforms(cfg) -> Tuple["RetinalPreprocessTransform", "RetinalPreprocessTransform"]:
    """Convenience factory returning (train_transform, eval_transform)."""
    train_tf = RetinalPreprocessTransform(
        image_size=cfg.image_size,
        train=True,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size,
    )
    eval_tf = RetinalPreprocessTransform(
        image_size=cfg.image_size,
        train=False,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size,
    )
    return train_tf, eval_tf
