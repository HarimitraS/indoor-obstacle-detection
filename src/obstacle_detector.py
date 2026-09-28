import cv2
import torch
import numpy as np

from model import IndoorObstacleModel


MODEL_PATH = "models/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Sofa",
    "Wall",
    "Pillar",
    "People",
    "Door",
    "Other",
    "Background"
]


model = IndoorObstacleModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


def preprocess(frame):

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    return tensor.unsqueeze(0).to(DEVICE)


def detect(frame):

    tensor = preprocess(frame)

    with torch.no_grad():

        output = model(tensor)

        output = torch.sigmoid(output)

    output = output[0].cpu().numpy()

    cells = output.reshape(
        30,
        7
    )

    return cells