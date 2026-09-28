import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from dataset import IndoorObstacleDataset
from model import IndoorObstacleModel


IMAGE_DIR = "dataset/raw/images"
LABEL_DIR = "dataset/raw/labels"

MODEL_DIR = "models"
MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 16
EPOCHS = 10

LEARNING_RATE = 0.001

SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", DEVICE)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


dataset = IndoorObstacleDataset(
    IMAGE_DIR,
    LABEL_DIR,
    (224, 224)
)


train_size = int(
    len(dataset) * 0.8
)

val_size = (
    len(dataset)
    - train_size
)


train_dataset, val_dataset = (
    random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator()
        .manual_seed(SEED)
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = IndoorObstacleModel()

model.to(DEVICE)


criterion = BCEWithLogitsLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for batch, (
        images,
        labels
    ) in enumerate(train_loader):

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()


        if batch % 50 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Batch {batch}/{len(train_loader)} "
                f"Loss {loss.item():.4f}",
                flush=True
            )


    train_loss /= len(
        train_loader
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()


    val_loss /= len(
        val_loader
    )


    print()
    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.5f}"
    )

    print(
        f"Val Loss:   {val_loss:.5f}"
    )


    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best model saved"
        )


print()
print("==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    "Best validation loss:",
    best_loss
)

print(
    "Model:",
    MODEL_PATH
)