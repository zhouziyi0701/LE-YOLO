import torch
import torch.nn as nn
from ultralytics.nn.modules import C3Ghost, CBAM

# 将C3Ghost和CBAM封装
class C3Ghost_CBAM(nn.Module):
    """
    C3Ghost block with a CBAM module appended at the end.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g1=1, e=0.5):
        """
        Initializes the C3Ghost_CBAM module.
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int, optional): Number of GhostBottleneck blocks. Defaults to 1.
            shortcut (bool, optional): Whether to use a shortcut connection. Defaults to True.
            g1 (int, optional): Group convolution parameter. Defaults to 1.
            e (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
        """
        super().__init__()
        self.c3ghost = C3Ghost(c1, c2, n, shortcut, g1, e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        x_out = self.c3ghost(x)
        return self.cbam(x_out)
