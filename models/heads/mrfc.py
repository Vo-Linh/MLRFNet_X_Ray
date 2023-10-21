import torch
import torch.nn as nn


class CSRA(nn.Module):
    """
    Channel-Spatial Residual Attention (CSRA) module.

    The CSRA module is a lightweight attention module 
    that can be used to improve the performance of convolutional neural networks. 
    It works by combining channel-wise and spatial-wise attention to generate a final attention map.

    Args:
        in_channels (list[int]): The number of input channels for the CSRA module.
        num_classes (int): The number of output classes for the CSRA module.
        T (int): The temperature for the softmax function.
        lam (int): The lambda parameter for the weighted sum of the base logit and the attention logit.

    Returns:
        nn.Module: The CSRA module.
    """

    def __init__(self, in_channels: list, num_classes: int, T: int, lam: int):
        super(CSRA, self).__init__()
        self.T = T      # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(in_channels, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        score = self.head(x) / torch.norm(self.head.weight,
                                          dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)
        
        score_soft = self.softmax(score * self.T)
        att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit


class MRFC(nn.Module):
    """
    Multi-Resolution Feature Channel (MRFC) module.

    The MRFC module is a lightweight attention module 
    that can be used to improve the performance of convolutional neural networks. 
    It works by combining channel-wise and spatial-wise attention 
    at multiple resolutions to generate a final attention map.
    This allows the MRFC module to capture global and local dependencies in the input features.

    Args:
        in_channels (list[int]): The number of input channels for the MRFC module.
        num_classes (int): The number of output classes for the MRFC module.
        lam (list[float]): The lambda parameter for the weighted sum of the base logit and the attention logit for each resolution.

    Returns:
        nn.Module: The MRFC module.

    >>> neck_feature = [torch.randn(1, 256, 32, 32), 
                        torch.randn(1, 512, 16, 16),
                        torch.randn(1, 1024, 8, 8)]
    >>> mrfc = MRFC(in_channels=[256, 512, 1024], 
                    num_classes=1000, 
                    lam=[0.25, 0.5, 0.25])
    >>> out = mrfc(neck_feature)
    >>> assert out.shape == (1, 1000)
    """

    def __init__(self, in_channels, num_classes, lam):
        super(MRFC, self).__init__()
        self.fpn_csra = nn.ModuleList()
        self.conv_1x1 = nn.ModuleList()
        for i in range(len(in_channels)):
            csra = CSRA(in_channels=in_channels[i],
                        num_classes=num_classes,
                        T=99,
                        lam=lam[i])
            conv_1x1 = nn.Conv2d(
                in_channels[i], in_channels[i], kernel_size=1, stride=1)
            self.conv_1x1.append(conv_1x1)
            self.fpn_csra.append(csra)

        self.sigmoid = nn.Sigmoid()

    def forward(self, neck_feature):
        """
        neck_feature(list[tensor]): feature got from necks
        """
        assert len(neck_feature) == len(self.fpn_csra)
        head_outs = []
        for i in range(len(self.fpn_csra)):
            out_conv = self.conv_1x1[i](neck_feature[i])
            out = self.fpn_csra[i](out_conv)
            head_outs.append(out)

        head_outs = torch.stack(head_outs, dim=1)
        p_sum = head_outs.sum(dim=1)
        
        return p_sum


if __name__ == "__main__":
    neck_feature = [torch.randn(1, 256, 32, 32),
                    torch.randn(1, 512, 16, 16),
                    torch.randn(1, 1024, 8, 8)]
    mrfc = MRFC(in_channels=[256, 512, 1024],
                num_classes=12,
                lam=[0.25, 0.5, 0.25])
    out = mrfc(neck_feature)
    print(out, out.shape)
