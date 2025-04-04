import torch
import torch.nn as nn

class CanopyHeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)  # R, G, B → altura

    def forward(self, x):
        avg = x.mean(dim=[2, 3])            # B x 3 (promedio de R, G, B)
        height = self.fc(avg)               # B x 1 (valor de altura)
        return height.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512, 512)  # B x 1 x 512 x 512
