import os
import shutil
import random

SRC = r"kaggle_download\extracted\dataset"
DST = r"data\test"

CLASS_MAP = {
    "cataract": "Cataracts",
    "diabetic_retinopathy": "Diabetic_Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
}

random.seed(42)

# Create fresh test directory
if os.path.exists(DST):
    shutil.rmtree(DST)

for source_class, target_class in CLASS_MAP.items():
    source_dir = os.path.join(SRC, source_class)
    target_dir = os.path.join(DST, target_class)

    os.makedirs(target_dir, exist_ok=True)

    images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]

    print(f"{source_class}: found {len(images)} images")

    if len(images) < 100:
        raise RuntimeError(
            f"Not enough images in {source_class}: {len(images)}"
        )

    selected = random.sample(images, 100)

    for filename in selected:
        shutil.copy2(
            os.path.join(source_dir, filename),
            os.path.join(target_dir, filename)
        )

    print(f"  -> copied 100 to {target_class}")

print("\nDONE: Dataset C = 400 images")