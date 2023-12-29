import torch
from torch import nn
import torch.nn.functional as F

class VIB(nn.Module):
    """
    Variational Inference Bottleneck (VIB) module for compressing feature maps.

    This module implements a single VIB block with encoder, mu, and std layers.
    The encoder compresses the input feature map, and the mu and std layers
    generate the mean and standard deviation of the latent distribution.

    Args:
        in_channels: Number of input channels.

    Attributes:
        encoder: Conv2d layer for compressing the input.
        mu: Conv2d layer for generating the mean of the latent distribution.
        std: Conv2d layer for generating the standard deviation of the latent distribution.
        relu: ReLU activation function.
    """

    def __init__(self, in_channel) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(in_channel, 2*in_channel, 1)
        self.mu = nn.Conv2d(2*in_channel, in_channel, 1)
        self.std = nn.Conv2d(2*in_channel, in_channel, 1)

    def forward(self, x):
        """
        Inputs:
        x: Tensor of shape (batch_size, in_channels, height, width).

        Outputs:
        z: Tensor of shape (batch_size, in_channels, height, width)
            representing the reparameterized latent variable."""
        encode = F.relu(self.encoder(x))
        mu = self.mu(encode)
        std = F.softplus(self.std(encode) - 5)

        eps = torch.randn_like(std)
        z = mu + std*eps

        return mu, std, z


class FPN_VIB(nn.Module):
    """
    Feature Pyramid Network with Variational Information Bottleneck (FPN_VIB).

    This class implements a Feature Pyramid Network with variational information bottleneck
    for handling multi-scale features.

    Args:
        in_channels (list[int]): Number of input channels per scale.

    Attributes:
        fpn_vib (nn.ModuleList): List of Variational Information Bottleneck (VIB) modules,
            one for each input scale.

    """

    def __init__(self, in_channels: list):
        super().__init__()
        self.fpn_vib = nn.ModuleList()

        for i in range(len(in_channels)):
            vib = VIB(in_channel=in_channels[i])
            self.fpn_vib.append(vib)

    def forward(self, mul_feature):
        """
        Forward pass of the FPN_VIB.

        Args:
            mul_feature (List[torch.Tensor]): List of input features from different scales.

        Returns:
            List[torch.Tensor]: List of output features after passing through the VIB modules.

        Raises:
            AssertionError: If the length of mul_feature is not equal to the number of scales.
        """
        assert len(mul_feature) == len(self.fpn_vib),\
            "Number of input scales does not match the expected number."
        vib_list = []
        mu_list = []
        std_list = []

        for i in range(len(self.fpn_vib)):
            mu, std, z = self.fpn_vib[i](mul_feature[i])
            vib_list.append(z)
            mu_list.append(mu)
            std_list.append(std)

        return mu_list, std_list, vib_list

if __name__ == "__main__":
    shape_in = [
        [1, 512, 28, 28], 
        [1, 1024, 14, 14],
        [1, 2048, 7, 7]
    ]
    backbone_feature = []
    for i in range(len(shape_in)):
        backbone_feature.append(torch.rand(shape_in[i]))
    
    vib_list = FPN_VIB(in_channels= [512, 1024, 2048])
    mu_out, std_out, vib_out = vib_list(backbone_feature)

    print(mu_out[0].shape, mu_out[1].shape, mu_out[2].shape)