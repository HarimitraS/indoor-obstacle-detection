import os

ROOT = os.path.abspath("dataset/raw")

print("\n========================================")
print("       DATASET INSPECTION")
print("========================================")

print("\nDataset location:")
print(ROOT)

if not os.path.exists(ROOT):
    print("\nERROR: Dataset folder does not exist!")
    exit()

print("\nTop-level contents:")

for item in os.listdir(ROOT):
    full_path = os.path.join(ROOT, item)

    if os.path.isdir(full_path):
        print(f"  [FOLDER] {item}")
    else:
        print(f"  [FILE]   {item}")

print("\n----------------------------------------")
print("Searching for image and label files...")
print("----------------------------------------")

image_count = 0
label_count = 0

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
label_extensions = (".txt", ".xml", ".json")

for root, dirs, files in os.walk(ROOT):

    for file in files:

        lower = file.lower()

        if lower.endswith(image_extensions):
            image_count += 1

        elif lower.endswith(label_extensions):
            label_count += 1

print(f"\nImages found : {image_count}")
print(f"Labels found : {label_count}")

print("\n========================================")
print("          INSPECTION COMPLETE")
print("========================================")