import os
import subprocess

DATASET = "sukai3316/indoor-obstacle-avoidance-dataset"
OUTPUT_DIR = "dataset/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("===================================")
print("   DOWNLOADING KAGGLE DATASET")
print("===================================")

command = [
    "kaggle",
    "datasets",
    "download",
    "-d",
    DATASET,
    "-p",
    OUTPUT_DIR,
    "--unzip"
]

result = subprocess.run(command)

if result.returncode == 0:
    print("\nDataset downloaded successfully!")
    print(f"Location: {OUTPUT_DIR}")
else:
    print("\nDataset download failed.")
    print("Check your Kaggle API credentials.")