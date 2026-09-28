import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


NUM_CELLS = 30
NUM_CLASSES = 7

OUTPUT_SIZE = NUM_CELLS * NUM_CLASSES


class IndoorObstacleModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        )

        self.backbone.classifier[1] = nn.Linear(
            1280,
            OUTPUT_SIZE
        )

    def forward(self, x):

        return self.backbone(x)


if __name__ == "__main__":

    model = IndoorObstacleModel()

    dummy = torch.randn(
        2,
        3,
        224,
        224
    )

    output = model(dummy)

    print("Input :", dummy.shape)
    print("Output:", output.shape)