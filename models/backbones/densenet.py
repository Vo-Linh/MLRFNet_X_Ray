import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Bottleneck(nn.Module):
    """
    A bottleneck block for DenseNets. It consists of two convolutional layers
    with a 1x1 kernel and 3x3 kernel, respectively, with a bottleneck structure
    for improved efficiency.

    Args:
        nChannels (int): Number of input channels.
        growthRate (int): Number of new channels added by each dense layer.
    """
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    """
    A basic dense layer for DenseNets. It consists of a single convolutional
    layer with a 3x3 kernel, followed by batch normalization and ReLU activation.

    Args:
        nChannels (int): Number of input channels.
        growthRate (int): Number of new channels added by the layer.
    """
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    """
    A transition block for DenseNets. It consists of a 1x1 convolutional layer
    followed by an average pooling layer, serving to reduce the number of
    channels and spatial dimensions between dense blocks.

    Args:
        nChannels (int): Number of input channels.
        nOutChannels (int): Number of output channels.
    """
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, bottleneck):
        """
        A DenseNet model as described in the paper "Densely Connected Convolutional
        Networks" by Huang et al. (2017). It features dense connectivity between
        layers, where each layer receives the feature maps of all preceding layers
        as input.

        Args:
            growthRate (int): Number of new channels added by each dense layer.
            depth (int): Total number of layers in the network.
            reduction (float): Compression factor for transition blocks.
            bottleneck (bool): Whether to use bottleneck layers.
        """
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out_1 = self.trans1(self.dense1(out))
        out_2 = self.trans2(self.dense2(out_1))
        out_3 = self.dense3(out_2)

        return [out_1, out_2, out_3]

if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    densenet = DenseNet(growthRate=32, depth=121, reduction=0.5, bottleneck=True)
    for i in range(3):
        print(densenet(x)[i].shape)
