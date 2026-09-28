"""
preprocessing.py
=================
Section 2: Custom Preprocessing & Augmentation Pipeline (REVISED).

Changes from the original version, based on train/test diagnostics:

  1. `crop_fundus_roi` now uses an ADAPTIVE (Otsu) threshold instead of a
     fixed one, with a safety check that skips cropping entirely when an
     image has no meaningful black border (already cropped). This fixes
     the class-correlated / dataset-correlated border artifact found by
     comparing avg_corner_brightness across train vs. test.

  2. Color is now PRESERVED. The old pipeline extracted only the green
     channel and replicated it into all 3 channels, discarding all color
     info -- risky for a 4-class problem where color cues matter (DR
     hemorrhages/exudates are red/yellow, cataract haze desaturates the
     whole image). CLAHE is now applied to the L-channel in LAB color
     space instead, which boosts local contrast without destroying hue.
     The old green-channel function is kept (unused) for reference/ablation.

  3. `RandomVerticalFlip` removed from augmentation (not anatomically
     meaningful for fundus images and doesn't help cross-camera
     generalization). `RandomAffine` now fills rotated corners with the
     dataset's mean pixel value instead of black, so it doesn't
     reintroduce the same "black border" confound the ROI-crop fix is
     trying to remove.

  4. Normalization now uses per-dataset computed mean/std (see
     `compute_dataset_stats.py`) instead of hardcoded ImageNet stats,
     since the post-CLAHE pixel distribution is not natural-photo-like.
     Falls back to ImageNet stats with a warning if no computed stats
     are available yet, so the pipeline still runs before you've
     generated them.

`RetinalPreprocessTransform` is a single callable usable as a torchvision
transform: it can be dropped straight into `transforms.Compose([...])` or
used stand-alone in a `Dataset.__getitem__`.
"""

import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


# --------------------------------------------------------------------- #
# Step 1: Circular fundus ROI auto-crop (ADAPTIVE)
# --------------------------------------------------------------------- #
def crop_fundus_roi(
    image_bgr: np.ndarray,
    min_background_fraction: float = 0.02,
) -> np.ndarray:
    """
    Crop tightly around the circular fundus region, discarding the black
    background that fundus cameras typically produce around the retina.

    Uses Otsu's method to pick the foreground/background threshold
    per-image instead of a single fixed value, so it adapts to different
    source datasets (different border darkness, different camera
    conventions) rather than behaving inconsistently across them.

    If an image has little or no dark border to begin with (already
    cropped by the dataset provider), this is detected and the image is
    returned unchanged -- this avoids Otsu incorrectly slicing into real
    fundus content on images that don't have a border in the first place.

    Args:
        image_bgr: HxWx3 uint8 image in BGR order (OpenCV convention).
        min_background_fraction: minimum fraction of the image that must
            be classified as "background" before we bother cropping.
            Below this, we assume the image is already tightly cropped.

    Returns:
        Cropped HxWx3 uint8 BGR image. If no meaningful border is found,
        or on any degenerate case, the original image is returned
        unchanged to avoid crashing the pipeline.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu picks the threshold automatically per-image.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's binary output doesn't know which side is "foreground" -- by
    # convention the fundus circle is the bright side and the border is
    # dark, so if more than half the image got marked as the mask, that
    # almost certainly means it inverted (bright border / dark center
    # doesn't happen in fundus photography), so flip it back.
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)

    background_fraction = 1.0 - (np.count_nonzero(mask) / mask.size)
    if background_fraction < min_background_fraction:
        # No meaningful black border present -- already cropped, skip.
        return image_bgr

    # Clean small noise specks so the bounding box isn't thrown off by
    # stray bright/dark pixels outside the actual fundus circle.
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
# Step 2a: LAB-space CLAHE (COLOR-PRESERVING -- now the default)
# --------------------------------------------------------------------- #
def apply_lab_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to the L (lightness) channel in LAB color space, then
    re-merge with the original a/b (color) channels. This boosts local
    contrast (vessels, lesions, optic disc rim) the same way the old
    green-channel approach did, but keeps hue/saturation intact -- so
    the network can still use color as a cue (red hemorrhages vs. yellow
    exudates vs. desaturated cataract haze), which a 4-class model needs.

    Args:
        image_bgr: HxWx3 uint8 BGR image.
        clip_limit: CLAHE contrast clipping threshold.
        tile_grid_size: CLAHE local neighborhood tile size.

    Returns:
        HxWx3 uint8 BGR image, contrast-enhanced, color preserved.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------- #
# Step 2b: Green-channel extraction + CLAHE (LEGACY -- kept for ablation)
# --------------------------------------------------------------------- #
def apply_green_channel_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Original approach: extract the green channel, apply CLAHE, re-project
    across all 3 channels. Discards color entirely. Kept only so you can
    still run the old pipeline as an ablation/comparison if you want to
    confirm the color-preserving switch actually helps.
    """
    green = image_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    green_eq = clahe.apply(green)
    return cv2.merge([green_eq, green_eq, green_eq])


# --------------------------------------------------------------------- #
# Combined callable transform
# --------------------------------------------------------------------- #
class RetinalPreprocessTransform:
    """
    End-to-end preprocessing + augmentation transform for retinal fundus
    images. Accepts a PIL.Image (RGB) and returns a normalized torch
    tensor ready for the model.

    Pipeline:
        PIL (RGB) -> BGR ndarray -> adaptive ROI crop -> LAB CLAHE
                   -> PIL -> resize -> [augmentations if train=True]
                   -> ToTensor -> Normalize(computed or ImageNet stats)
    """

    # Fallback only -- prefer passing norm_mean/norm_std computed from
    # your own data via compute_dataset_stats.py.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_size: int = 300,
        train: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        use_lab_clahe: bool = True,
        norm_mean: Optional[list] = None,
        norm_std: Optional[list] = None,
    ):
        self.image_size = image_size
        self.train = train
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.use_lab_clahe = use_lab_clahe

        if norm_mean is None or norm_std is None:
            warnings.warn(
                "RetinalPreprocessTransform: no computed norm_mean/norm_std "
                "provided -- falling back to ImageNet stats. Run "
                "compute_dataset_stats.py and pass the result via Config "
                "for accurate normalization.",
                stacklevel=2,
            )
            norm_mean = self.IMAGENET_MEAN
            norm_std = self.IMAGENET_STD

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Fill color for RandomAffine's rotated corners, in 0-255 range,
        # matching the dataset's actual mean pixel value instead of
        # defaulting to black (which would reintroduce a border-like
        # artifact right after we just fixed the ROI-crop one).
        fill_255 = tuple(int(round(m * 255)) for m in norm_mean)

        # Geometric + color augmentations, applied only during training.
        # NOTE: RandomVerticalFlip removed -- not anatomically meaningful
        # for fundus images and adds noise without helping cross-camera
        # generalization.
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, fill=fill_255),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])

        self.to_tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])

    def __call__(self, pil_image: Image.Image) -> torch.Tensor:
        # 1. PIL RGB -> OpenCV BGR ndarray
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 2. Adaptive ROI crop
        bgr = crop_fundus_roi(bgr)

        # 3. Contrast enhancement (color-preserving by default)
        if self.use_lab_clahe:
            bgr = apply_lab_clahe(
                bgr,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid_size,
            )
        else:
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
    """
    Convenience factory returning (train_transform, eval_transform).

    Reads cfg.norm_mean / cfg.norm_std if present (populated automatically
    by Config.__post_init__ from compute_dataset_stats.py's output, if
    that file exists). Falls back to ImageNet stats with a warning
    otherwise -- run compute_dataset_stats.py once and re-run training
    for correct normalization.
    """
    norm_mean = getattr(cfg, "norm_mean", None)
    norm_std = getattr(cfg, "norm_std", None)
    use_lab_clahe = getattr(cfg, "use_lab_clahe", True)

    train_tf = RetinalPreprocessTransform(
        image_size=cfg.image_size,
        train=True,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size,
        use_lab_clahe=use_lab_clahe,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    eval_tf = RetinalPreprocessTransform(
        image_size=cfg.image_size,
        train=False,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_grid_size=cfg.clahe_tile_grid_size,
        use_lab_clahe=use_lab_clahe,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    return train_tf, eval_tf