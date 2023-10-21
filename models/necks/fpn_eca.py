import torch
from torch import nn


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = torch.sigmoid(y)

        return x * y.expand_as(x)


class FPN_ECA(nn.Module):
    """
    Args:
        in_channels (list[int]): Number of input channels per scale."""

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.fpn_eca = nn.ModuleList()

        for i in range(len(in_channels)):
            eca = ECA(channel=in_channels[i])
            self.fpn_eca.append(eca)

    def forward(self, mul_feature):
        """
        mul_feature(list[tensor]): feature got from backbones
        """
        assert len(mul_feature) == len(self.fpn_eca)
        neck_feature = []
        for i in range(len(self.fpn_eca)):
            out = self.fpn_eca[i](mul_feature[i])
            neck_feature.append(out)

        return neck_feature

if __name__ == "__main__":
    shape_in = [
        [1, 512, 28, 28], 
        [1, 1024, 14, 14],
        [1, 2048, 7, 7]
    ]
    backbone_feature = []
    for i in range(len(shape_in)):
        backbone_feature.append(torch.rand(shape_in[i]))
    
    fpn_eca = FPN_ECA(in_channels= [512, 1024, 2048])
    neck_out = fpn_eca(backbone_feature)

    print(neck_out[0].shape)
