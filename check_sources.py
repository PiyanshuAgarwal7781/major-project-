"""
check_sources.py
Prints, for every class in train and test:
  - how many images, and the most common image sizes (a fingerprint of the source camera/dataset)
  - the average RGB colour of the retina area (a fingerprint of colour cast)
Run:  python check_sources.py
"""
import os
import random
import collections

import numpy as np
from PIL import Image

from config import CFG

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def mean_colour(paths, n=60):
    vals = []
    for p in random.sample(paths, min(n, len(paths))):
        im = Image.open(p)
        try:
            im.draft("RGB", (128, 128))
        except Exception:
            pass
        a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=float)
        mask = a.sum(2) > 30                      # ignore black border
        if mask.any():
            vals.append(a[mask].mean(0))
    return np.mean(vals, 0).round(0).astype(int).tolist() if vals else None


for name, root in (("TRAIN", CFG.train_root), ("TEST", CFG.test_root)):
    print(f"\n=== {name}  ({root}) ===")
    for c in CFG.class_names:
        d = os.path.join(root, c)
        paths = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXT)]
        sizes = collections.Counter()
        for p in paths:
            with Image.open(p) as im:
                sizes[im.size] += 1
        print(f"{c:22s} n={len(paths):5d}  mean RGB={mean_colour(paths)}")
        print(f"{'':22s} top sizes: {sizes.most_common(4)}")