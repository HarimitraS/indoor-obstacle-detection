import os
import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import IndoorObstacleDataset
from model import IndoorObstacleModel


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = "dataset/raw/images"
LABEL_DIR = "dataset/raw/labels"

MODEL_PATH = "models/best_model.pth"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

TRAIN_RATIO = 0.8

SEED = 42

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

dataset = IndoorObstacleDataset(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    image_size=IMAGE_SIZE
)

# Same split as training
train_size = int(
    TRAIN_RATIO * len(dataset)
)

val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = IndoorObstacleModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= THRESHOLD
        ).int()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

y_pred = np.array(
    all_predictions
).flatten()

y_true = np.array(
    all_labels
).flatten()


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\n")
print("=" * 60)


# ============================================================
# POSITIVE / NEGATIVE COUNTS
# ============================================================

print(
    "\nActual positive labels:",
    int(y_true.sum())
)

print(
    "Predicted positive labels:",
    int(y_pred.sum())
)

print(
    "Total label values:",
    len(y_true)
)