"""
dataset.py
==========
Dataset + split utilities for Retinal-SwinNet.

Expected on-disk layout (standard ImageFolder-style):

    data_root/
        Glaucoma/                    *.jpg / *.png
        Cataracts/
        Diabetic_Retinopathy/
        Normal/

The training directory contains Dataset A + Dataset B combined.

REVISED (split): the merged dataset is now split into train / val / test
by `split_dataset.py`, which writes CSV manifests to `data/splits/`.
`ManifestDataset` + `make_subsets_from_manifest` load those manifests.
The test split must NEVER be used for training, validation,
hyperparameter tuning, or model selection -- only for final evaluation.

(The old `data/test` folder / `make_subsets` / `stratified_split` are kept
below for reference but are no longer used by train.py.)

REVISED: some source datasets ship with baked-in offline-augmented
duplicates alongside the originals (filenames like aug_hflip_..,
aug_rotate_.., aug_noise_.., aug_clahe_..). These are near-identical
copies of the same underlying photo. Left in place, a stratified
file-level split can put one variant in train and another variant of
the SAME source image in val -- the model then "validates well" on
images it has effectively already seen, which doesn't reflect real
generalization (this is very likely why val accuracy looked fine while
accuracy on the truly independent test set collapsed).

By default, RetinalFundusDataset now EXCLUDES any filename matching a
known "aug_*" prefix, so only original captured images are used for
training/validation. Your own RetinalPreprocessTransform already
applies live flip/rotation/CLAHE augmentation every epoch, so nothing
is lost by dropping the baked-in copies -- if anything, this removes a
source of double/inconsistent augmentation. Set
`exclude_augmented=False` to restore the old behavior (e.g. for an
ablation comparing with/without this filter).
"""

import csv
import os
import random
import re
from collections import Counter
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)

# Matches baked-in offline-augmented filenames like:
#   aug_hflip_042_Glaucoma.png
#   aug_rotate_7_Normal.jpg
#   aug_noise_113_Cataracts.png
#   aug_clahe_5_Diabetic_Retinopathy.png
AUGMENTED_FILENAME_PATTERN = re.compile(
    r"^aug_(clahe|hflip|noise|rotate)_",
    re.IGNORECASE,
)


class RetinalFundusDataset(Dataset):
    """Loads (image_path, label) pairs from a class-per-folder directory tree."""

    def __init__(
        self,
        data_root: str,
        class_names: List[str],
        transform=None,
        exclude_augmented: bool = True,
    ):
        self.data_root = data_root
        self.class_names = class_names
        self.transform = transform
        self.exclude_augmented = exclude_augmented
        self.samples: List[Tuple[str, int]] = []

        n_skipped_augmented = 0

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_root, class_name)

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}. "
                    f"Check that data_root points to a directory containing "
                    f"one subfolder per class in {class_names}."
                )

            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(IMG_EXTENSIONS):
                    continue

                if self.exclude_augmented and AUGMENTED_FILENAME_PATTERN.match(fname):
                    n_skipped_augmented += 1
                    continue

                self.samples.append(
                    (
                        os.path.join(class_dir, fname),
                        label_idx,
                    )
                )

        if self.exclude_augmented:
            print(
                f"[Dataset] {data_root}: excluded {n_skipped_augmented} "
                f"baked-in augmented files (aug_hflip_/aug_rotate_/"
                f"aug_noise_/aug_clahe_), kept {len(self.samples)} originals."
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {data_root}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, path

    def class_distribution(self) -> Counter:
        """Return number of samples belonging to each class."""
        return Counter(label for _, label in self.samples)


class ManifestDataset(RetinalFundusDataset):
    """
    Loads samples from a split CSV written by split_dataset.py.

    CSV columns: path (relative to data_root), class, label, group_id.
    The label is recomputed from `class_names` so it always matches
    CFG.class_names ordering.
    """

    def __init__(
        self,
        data_root: str,
        manifest_csv: str,
        class_names: List[str],
        transform=None,
    ):
        self.data_root = data_root
        self.class_names = class_names
        self.transform = transform
        self.exclude_augmented = False
        self.samples: List[Tuple[str, int]] = []

        if not os.path.isfile(manifest_csv):
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_csv}. "
                f"Run `python split_dataset.py` first."
            )

        cls_to_idx = {c: i for i, c in enumerate(class_names)}

        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append(
                    (
                        os.path.join(data_root, row["path"]),
                        cls_to_idx[row["class"]],
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples in {manifest_csv}.")


def make_subsets_from_manifest(
    train_root: str,
    splits_dir: str,
    class_names: List[str],
    train_transform,
    eval_transform,
):
    """
    Build train / val / test datasets from the CSV manifests produced by
    split_dataset.py (a leakage-aware, stratified split of the merged
    Dataset A + B).

        train -> train_transform (with augmentation)
        val   -> eval_transform  (used for model / checkpoint selection)
        test  -> eval_transform  (final evaluation ONLY)
    """
    def load(name, tf):
        return ManifestDataset(
            train_root,
            os.path.join(splits_dir, f"{name}.csv"),
            class_names,
            transform=tf,
        )

    return (
        load("train", train_transform),
        load("val", eval_transform),
        load("test", eval_transform),
    )


# ---------------------------------------------------------------------- #
# LEGACY (not used by train.py anymore)
# ---------------------------------------------------------------------- #
def stratified_split(
    dataset: RetinalFundusDataset,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Legacy file-level stratified split. Superseded by split_dataset.py,
    which additionally groups near-duplicate images so they never straddle
    splits.
    """

    rng = random.Random(seed)

    by_class = {}

    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    train_idx = []
    val_idx = []
    test_idx = []

    for label, indices in by_class.items():

        rng.shuffle(indices)

        n = len(indices)

        n_val = max(
            1,
            int(round(n * val_split))
        )

        n_test = (
            0
            if test_split <= 0
            else max(1, int(round(n * test_split)))
        )

        val_idx.extend(
            indices[:n_val]
        )

        test_idx.extend(
            indices[n_val:n_val + n_test]
        )

        train_idx.extend(
            indices[n_val + n_test:]
        )

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
    exclude_augmented: bool = True,
):
    """
    LEGACY: train/val from the merged folder + separate Dataset C as test.
    Superseded by make_subsets_from_manifest.
    """

    base_train = RetinalFundusDataset(
        train_root,
        class_names,
        transform=train_transform,
        exclude_augmented=exclude_augmented,
    )

    base_eval = RetinalFundusDataset(
        train_root,
        class_names,
        transform=eval_transform,
        exclude_augmented=exclude_augmented,
    )

    independent_test = RetinalFundusDataset(
        test_root,
        class_names,
        transform=eval_transform,
        exclude_augmented=False,
    )

    train_idx, val_idx, _ = stratified_split(
        base_train,
        val_split=val_split,
        test_split=0.0,
        seed=seed,
    )

    train_set = Subset(
        base_train,
        train_idx,
    )

    val_set = Subset(
        base_eval,
        val_idx,
    )

    return train_set, val_set, independent_test


def compute_class_weights(
    dataset: RetinalFundusDataset,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    These weights are used by the Class-Weighted Focal Loss.
    Pass the TRAIN split only.
    """

    counts = dataset.class_distribution()

    total = sum(counts.values())

    weights = torch.zeros(
        num_classes,
        dtype=torch.float32,
    )

    for c in range(num_classes):

        n_c = counts.get(c, 1)

        weights[c] = (
            total /
            (num_classes * n_c)
        )

    return weights