import os
import glob
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class IndoorObstacleDataset(Dataset):

    def __init__(self, image_dir, label_dir, image_size=(224, 224)):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size

        self.image_files = sorted(
            glob.glob(
                os.path.join(image_dir, "*.png")
            )
        )

        print(f"Images found: {len(self.image_files)}")

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {image_dir}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        # ------------------------------------------------
        # Load image
        # ------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        image = image.resize(self.image_size)

        image = np.array(image, dtype=np.float32)

        # Convert:
        # H x W x C
        #
        # to:
        # C x H x W

        image = image.transpose(2, 0, 1)

        # Normalize 0-255 → 0-1

        image = image / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # ------------------------------------------------
        # Load label
        # ------------------------------------------------

        base_name = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        label_path = os.path.join(
            self.label_dir,
            base_name + ".txt"
        )

        if not os.path.exists(label_path):

            raise FileNotFoundError(
                f"Label not found:\n{label_path}"
            )

        with open(label_path, "r") as file:

            values = [
                int(x)
                for x in file.read().split()
            ]

        # We expect:
        #
        # 30 Cells
        # ×
        # 7 values
        #
        # = 210

        if len(values) != 210:

            raise ValueError(
                f"Expected 210 label values, "
                f"but found {len(values)} "
                f"in {label_path}"
            )

        label = np.array(
            values,
            dtype=np.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


if __name__ == "__main__":

    dataset = IndoorObstacleDataset(
        image_dir="dataset/raw/images",
        label_dir="dataset/raw/labels"
    )

    print("\nDataset test")
    print("----------------------------")

    image, label = dataset[0]

    print("Image shape :", image.shape)
    print("Label shape :", label.shape)

    print("Label values:", label)

    print("\nDataset loading successfully!")