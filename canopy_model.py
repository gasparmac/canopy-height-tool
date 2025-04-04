import torch
import torch.nn as nn
import torchvision.models as models

class CanopyHeightNet(nn.Module):
    def __init__(self):
        super(CanopyHeightNet, self).__init__()
        backbone = models.resnet50(weights=None)
        layers = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.regressor = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x.squeeze(1)
