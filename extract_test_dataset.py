from datasets import load_dataset
from pathlib import Path

print("Loading dataset...")
ds = load_dataset("AlexeyGHT/Fundus_CFP")
data = ds["train"]

output = Path("data/test")

classes = {
    "Glaucoma": (0, 1000),
    "Cataracts": (1000, 2000),
    "Normal": (2000, 3000),
    "Diabetic_Retinopathy": (6000, 7000),
}

for class_name, (start, end) in classes.items():
    class_dir = output / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {class_name}: {start}-{end-1}")

    for i in range(start, end):
        image = data[i]["image"]
        image.save(class_dir / f"{class_name}_{i:04d}.jpg")

    print(f"Saved {end - start} images.")

print("\nDONE")
print("Dataset C created at:", output.resolve())

for class_name in classes:
    count = len(list((output / class_name).glob("*.jpg")))
    print(f"{class_name}: {count}")