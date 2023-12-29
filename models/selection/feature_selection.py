import torch
from torch import nn

class FeatureSelection(nn.Module):
    """Constructs a Feature Selection module.
    Args:
        channel: Number of channels of the input feature map
    """
    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channels = 1,
                            kernel_size = 1, stride = 1)

    def forward(self, x):
        x_conv = self.conv(x)
        x_out = torch.sigmoid(x_conv)

        return x_out

class FPN_FeatureSelection(nn.Module):
    """
    Args:
        in_channels (list[int]): Number of input channels per scale."""

    def __init__(self, in_channels: list):
        super().__init__()
        self.fpn_fs = nn.ModuleList()

        for i in range(len(in_channels)):
            fs = FeatureSelection(in_channel=in_channels[i])
            self.fpn_fs.append(fs)

    def forward(self, mul_feature):
        """
        mul_feature(list[tensor]): feature got from backbones
        """
        assert len(mul_feature) == len(self.fpn_fs)
        fs_list = []
        for i in range(len(self.fpn_fs)):
            out = self.fpn_fs[i](mul_feature[i])
            fs_list.append(out)

        return fs_list

if __name__ == "__main__":
    shape_in = [
        [1, 512, 28, 28], 
        [1, 1024, 14, 14],
        [1, 2048, 7, 7]
    ]
    backbone_feature = []
    for i in range(len(shape_in)):
        backbone_feature.append(torch.rand(shape_in[i]))
    
    fs_list = FPN_FeatureSelection(in_channels= [512, 1024, 2048])
    fs_out = fs_list(backbone_feature)

    print(fs_out[0].shape, fs_out[0])