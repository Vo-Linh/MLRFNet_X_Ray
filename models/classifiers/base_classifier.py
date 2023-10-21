import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    """Base Class for classifier
    
    Combine config from backbone, neck and head.
    """
    def __init__(self,
                 backbone,
                 neck,
                 head) -> None:
        super().__init__()