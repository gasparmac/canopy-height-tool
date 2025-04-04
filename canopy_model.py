import torch
import torch.nn as nn

class CanopyHeightNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: B x 3 x 512 x 512
        r = x[:, 0, :, :]  # canal rojo
        g = x[:, 1, :, :]  # canal verde
        b = x[:, 2, :, :]  # canal azul

        # Simulamos un mapa de altura en función del color
        height = 0.3 * r + 0.6 * g + 0.1 * b  # combinación ponderada
        return height  # B x 512 x 512
