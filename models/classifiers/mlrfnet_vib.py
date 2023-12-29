from torch import nn

class MLRFNet_VIB(nn.Module):
    """Combine MLRFNet module with VIB
    
    """
    def __init__(self, backbone, neck, head, vib) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.vib = vib


    def forward(self, x):
        """
        x --> bacbone --> VIB --> Neck  ---> Head --> Out
        """
        out_backbone = self.backbone(x)
        mu, std, vib = self.vib(out_backbone)

        out_neck = self.neck(vib)
        out = self.head(out_neck)

        return out, mu, std