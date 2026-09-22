"""
dataset.py
==========
Dataset + split utilities for Retinal-SwinNet.

Expected on-disk layout (standard ImageFolder-style):

    data_root/
        Glaucoma/               *.jpg / *.png
        Cataracts/
        Diabetic_Retinopathy/
        AMD/
        Normal/

This keeps things compatible with most public fundus datasets (ODIR,
APTOS-derived multi-class recompilations, etc.) after they're re-sorted
into per-class folders.
"""

import os
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class RetinalFundusDataset(Dataset):
    """Loads (image_path, label) pairs from a class-per-folder directory tree."""

    def __init__(self, data_root: str, class_names: List[str], transform=None):
        self.data_root = data_root
        self.class_names = class_names
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}. "
                    f"Check that data_root points at a directory containing "
                    f"one subfolder per class in {class_names}."
                )
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {data_root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, path

    def class_distribution(self) -> Counter:
        return Counter(label for _, label in self.samples)


def stratified_split(
    dataset: RetinalFundusDataset,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified train/val/test split so each split preserves the overall
    class distribution -- important given the class imbalance called out
    in Section 4 (e.g. fewer AMD samples than Normal).
    """
    rng = random.Random(seed)
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in by_class.items():
        rng.shuffle(indices)
        n = len(indices)
        n_val = max(1, int(round(n * val_split)))
        n_test = 0 if test_split <= 0 else max(1, int(round(n * test_split)))
        val_idx.extend(indices[:n_val])
        test_idx.extend(indices[n_val:n_val + n_test])
        train_idx.extend(indices[n_val + n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def make_subsets(
    train_root: str,
    test_root: str,
    class_names: List[str],
    train_transform,
    eval_transform,
    val_split: float,
    seed: int = 42,
):
    """
    Creates train/validation subsets from the combined training datasets
    and loads the independent third dataset as the final test set.

    Dataset C is never used for training, validation, or model selection.
    """
    base_train = RetinalFundusDataset(
        train_root, class_names, transform=train_transform
    )
    base_eval = RetinalFundusDataset(
        train_root, class_names, transform=eval_transform
    )
    independent_test = RetinalFundusDataset(
        test_root, class_names, transform=eval_transform
    )

    train_idx, val_idx, _ = stratified_split(
        base_train, val_split=val_split, test_split=0.0, seed=seed
    )

    train_set = Subset(base_train, train_idx)
    val_set = Subset(base_eval, val_idx)
    return train_set, val_set, independent_test


def compute_class_weights(dataset: RetinalFundusDataset, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for the loss function
    (Section 4: Class-Weighted Focal Loss).
    """
    counts = dataset.class_distribution()
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n_c = counts.get(c, 1)  # avoid div-by-zero for an absent class
        weights[c] = total / (num_classes * n_c)
    return weights
