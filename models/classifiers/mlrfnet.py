import torch
from torch import nn

class MLRFNet(nn.Module):
    def __init__(self, backbone, neck, head) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        layer = [self.backbone, self.neck, self.head]
        self.net = nn.Sequential(*layer)


    def forward(self, x):
        out = x
        for model in self.net:
            out = model(out)
            
        return out
