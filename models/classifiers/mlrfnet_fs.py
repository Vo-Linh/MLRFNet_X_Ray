import torch
from torch import nn

class MLRFNet_FS(nn.Module):
    """Combine MLRFNet module with feature selection
    
    """
    def __init__(self, backbone, neck, head, feature_selection) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.feature_selection = feature_selection


    def forward(self, x):
        """
        x --> bacbone --> FS 
                |         |
                |-------> X --> Neck  ---> Head --> Out
        """
        out_backbone = self.backbone(x)
        feature_selection = self.feature_selection(out_backbone)

        out_feature_selection = []

        for i in range(len(feature_selection)):
            out_feature_selection.append(feature_selection[i] * out_backbone[i])

        out_neck = self.neck(out_feature_selection)
        out = self.head(out_neck)

        return feature_selection, out

