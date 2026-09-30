"""
split_dataset.py
Leakage-aware, stratified train/val/test split of the merged dataset (data/train).
Writes CSV manifests to data/splits/. Images are NOT moved or copied.
Never touches data/test.

    python split_dataset.py          # 80/10/10
    python split_dataset.py --force  # overwrite (changes the test set!)
"""
import argparse
import csv
import json
import os
import re
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from config import CFG
from dataset import RetinalFundusDataset

HASH_SIZE = 16                     # 16x16 dHash -> 256 bits
HASH_BITS = HASH_SIZE * HASH_SIZE


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def dhash_pair(path):
    """dHash of the image and of its horizontal flip (catches flipped copies)."""
    with Image.open(path) as im:
        try:
            im.draft("RGB", (256, 256))
        except Exception:
            pass
        g = im.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    h = (a[:, 1:] > a[:, :-1]).flatten()
    af = a[:, ::-1]
    hf = (af[:, 1:] > af[:, :-1]).flatten()
    return h.astype(np.uint8), hf.astype(np.uint8)


def near_duplicate_pairs(H, Hf, threshold, chunk=512):
    Hn = H.astype(np.float32)
    Hfn = Hf.astype(np.float32)
    sums = Hn.sum(1)
    sums_f = Hfn.sum(1)
    n = len(H)
    for s in range(0, n, chunk):
        a = Hn[s:s + chunk]
        sa = a.sum(1)[:, None]
        d = sa + sums[None, :] - 2 * (a @ Hn.T)          # hamming vs originals
        df = sa + sums_f[None, :] - 2 * (a @ Hfn.T)      # hamming vs flipped
        dmin = np.minimum(d, df)
        ii, jj = np.where(dmin <= threshold)
        for i, j in zip(ii, jj):
            if j > s + i:
                yield s + i, j


def build_groups(paths, threshold, patient_regex):
    n = len(paths)
    dsu = DSU(n)

    print(f"[Split] hashing {n} images...")
    H = np.zeros((n, HASH_BITS), dtype=np.uint8)
    Hf = np.zeros((n, HASH_BITS), dtype=np.uint8)
    for i, p in enumerate(paths):
        H[i], Hf[i] = dhash_pair(p)
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{n}")

    n_edges = 0
    for i, j in near_duplicate_pairs(H, Hf, threshold):
        dsu.union(i, j)
        n_edges += 1
    print(f"[Split] near-duplicate pairs found (hamming <= {threshold}/{HASH_BITS}): {n_edges}")

    if patient_regex:
        rx = re.compile(patient_regex, re.IGNORECASE)
        seen, matched = {}, 0
        for i, p in enumerate(paths):
            m = rx.match(os.path.basename(p))
            if m:
                matched += 1
                key = m.group(1)
                if key in seen:
                    dsu.union(i, seen[key])
                else:
                    seen[key] = i
        print(f"[Split] patient-id regex matched {matched}/{n} filenames")

    roots = np.array([dsu.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    sizes = Counter(groups.tolist())
    multi = [s for s in sizes.values() if s > 1]
    print(f"[Split] {len(sizes)} groups | groups with >1 image: {len(multi)} | "
          f"largest group: {max(sizes.values())}")
    return groups


def one_fold(idx, labels, groups, n_splits, seed):
    """Hold out one stratified, group-respecting fold. Returns (rest, held)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, held = next(sgkf.split(idx, labels[idx], groups[idx]))
    return idx[tr], idx[held]


def write_csv(path, idx, rel_paths, labels, groups, class_names):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "label", "group_id"])
        for i in sorted(idx):
            w.writerow([rel_paths[i], class_names[labels[i]], int(labels[i]), int(groups[i])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default=CFG.train_root)
    ap.add_argument("--splits_dir", default=CFG.splits_dir)
    ap.add_argument("--val_frac", type=float, default=CFG.split_val_frac)
    ap.add_argument("--test_frac", type=float, default=CFG.split_test_frac)
    ap.add_argument("--hash_threshold", type=int, default=CFG.split_hash_threshold)
    ap.add_argument("--patient_regex", default=r"^(\d+)_(?:left|right)",
                    help="regex with ONE capture group = patient id; '' to disable")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.splits_dir, exist_ok=True)
    out = {k: os.path.join(args.splits_dir, f"{k}.csv") for k in ("train", "val", "test")}
    if not args.force and any(os.path.exists(p) for p in out.values()):
        raise SystemExit(
            f"Splits already exist in {args.splits_dir}. Re-splitting changes the test set; "
            f"pass --force only if you really want that."
        )

    # Same enumeration as training (aug_* files excluded, data/test never read).
    ds = RetinalFundusDataset(args.train_root, CFG.class_names, exclude_augmented=True)
    paths = [p for p, _ in ds.samples]
    labels = np.array([l for _, l in ds.samples])
    rel_paths = [os.path.relpath(p, args.train_root).replace("\\", "/") for p in paths]
    n = len(paths)

    groups = build_groups(paths, args.hash_threshold, args.patient_regex or None)

    n_test_folds = max(2, round(1 / args.test_frac))
    n_val_folds = max(2, round(1 / (args.val_frac / (1 - args.test_frac))))
    all_idx = np.arange(n)
    rest, test_idx = one_fold(all_idx, labels, groups, n_test_folds, args.seed)
    train_idx, val_idx = one_fold(rest, labels, groups, n_val_folds, args.seed + 1)

    # Leakage check: no group may appear in more than one split.
    gs = {k: set(groups[v].tolist()) for k, v in
          (("train", train_idx), ("val", val_idx), ("test", test_idx))}
    assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"]) \
        and not (gs["val"] & gs["test"]), "group leakage across splits!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    for k, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        write_csv(out[k], idx, rel_paths, labels, groups, CFG.class_names)

    print(f"\n{'Class':<24}" + "".join(f"{k:>16}" for k in ("train", "val", "test")))
    for ci, cname in enumerate(CFG.class_names):
        row = f"{cname:<24}"
        for idx in (train_idx, val_idx, test_idx):
            c = int((labels[idx] == ci).sum())
            row += f"{c:>7} ({100 * c / len(idx):4.1f}%)"
        print(row)
    print(f"{'TOTAL':<24}" + "".join(
        f"{len(i):>7} ({100 * len(i) / n:4.1f}%)" for i in (train_idx, val_idx, test_idx)))

    with open(os.path.join(args.splits_dir, "split_meta.json"), "w") as f:
        json.dump({
            "seed": args.seed, "n_images": n,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "hash_threshold": args.hash_threshold, "patient_regex": args.patient_regex,
            "n_groups": int(groups.max() + 1),
        }, f, indent=2)
    print(f"\n[Split] wrote manifests to {args.splits_dir}")


if __name__ == "__main__":
    main()