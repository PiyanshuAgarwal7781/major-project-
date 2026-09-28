#  python diagnose_datasets.py
import os, cv2, numpy as np

def scan(root):
    stats = {}
    for cls in os.listdir(root):
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        sizes, means, border_darkness = [], [], []
        for f in os.listdir(d)[:50]:  # sample 50 per class
            img = cv2.imread(os.path.join(d, f))
            if img is None: continue
            sizes.append(img.shape[:2])
            means.append(img.reshape(-1,3).mean(axis=0))
            corner = img[0:20, 0:20].mean()
            border_darkness.append(corner)
        stats[cls] = {
            "n_sampled": len(sizes),
            "avg_size": np.mean(sizes, axis=0) if sizes else None,
            "avg_rgb_mean": np.mean(means, axis=0) if means else None,
            "avg_corner_brightness": np.mean(border_darkness) if border_darkness else None,
        }
    return stats

print("=== TRAIN (Dataset A+B) ===")
for k,v in scan("./data/train").items(): print(k, v)
print("\n=== TEST (Dataset C, unseen) ===")
for k,v in scan("./data/test").items(): print(k, v)