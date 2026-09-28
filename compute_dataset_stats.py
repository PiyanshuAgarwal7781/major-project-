"""
compute_dataset_stats.py
=========================
Run this ONCE (and again any time you change what's in data/train/, e.g.
after swapping in new cataract images) before training.

Computes per-channel mean/std over the TRAINING set, measured AFTER the
same ROI-crop + CLAHE steps used at train time but BEFORE normalization --
this is the correct way to get normalization stats, since we need the
statistics of what the network will actually see, not of the raw files
on disk.

Writes the result to Config.stats_path (default: ./outputs/dataset_stats.json).
config.py automatically loads this file on the next run, so
preprocessing.py will stop falling back to (mismatched) ImageNet stats.

Usage:python compute_dataset_stats.py
"""

import json
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import CFG
from preprocessing import apply_green_channel_clahe, apply_lab_clahe, crop_fundus_roi


def compute_stats(data_root: str, class_names, image_size: int, use_lab_clahe: bool):
    running_sum = np.zeros(3, dtype=np.float64)
    running_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    paths = []
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        if not os.path.isdir(class_dir):
            print(f"  [skip] {class_dir} not found")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                paths.append(os.path.join(class_dir, fname))

    print(f"Computing stats over {len(paths)} training images...")

    for path in tqdm(paths):
        pil_image = Image.open(path).convert("RGB")
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        bgr = crop_fundus_roi(bgr)

        if use_lab_clahe:
            bgr = apply_lab_clahe(
                bgr,
                clip_limit=CFG.clahe_clip_limit,
                tile_grid_size=CFG.clahe_tile_grid_size,
            )
        else:
            bgr = apply_green_channel_clahe(
                bgr,
                clip_limit=CFG.clahe_clip_limit,
                tile_grid_size=CFG.clahe_tile_grid_size,
            )

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

        arr = rgb.astype(np.float64) / 255.0  # match ToTensor's 0-1 scaling
        pixels = arr.reshape(-1, 3)

        running_sum += pixels.sum(axis=0)
        running_sq_sum += (pixels ** 2).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = running_sum / pixel_count
    variance = (running_sq_sum / pixel_count) - (mean ** 2)
    std = np.sqrt(np.clip(variance, a_min=1e-8, a_max=None))

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    mean, std = compute_stats(
        data_root=CFG.train_root,
        class_names=CFG.class_names,
        image_size=CFG.image_size,
        use_lab_clahe=CFG.use_lab_clahe,
    )

    print("\nComputed normalization stats (RGB order):")
    print(f"  mean = {mean}")
    print(f"  std  = {std}")

    os.makedirs(os.path.dirname(CFG.stats_path) or ".", exist_ok=True)
    with open(CFG.stats_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    print(f"\nSaved to {CFG.stats_path}")
    print("These will be auto-loaded by Config on your next training run.")