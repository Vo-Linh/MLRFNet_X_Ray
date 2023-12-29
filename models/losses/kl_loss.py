import torch
from torch import nn

def kl_loss(mu: torch.Tensor,
             std: torch.Tensor,
             loss_weight: float = 0.1,
             reduction_override: str =None) -> torch.Tensor:
    """
    Computes the Kullback-Leibler (KL) divergence loss between a given Gaussian distribution
    with mean `mu` and standard deviation `std` and a standard normal distribution.

    Args:
        mu (torch.Tensor): Mean of the Gaussian distribution.
        std (torch.Tensor): Standard deviation of the Gaussian distribution.
        loss_weight (float, optional): Weight for the KL loss. Default is 0.1.
        reduction_override (str, optional): Specifies the reduction method for the loss.
            Should be one of [None, 'none', 'mean', 'sum']. Default is None.

    Returns:
        torch.Tensor: KL divergence loss.

    Raises:
        AssertionError: If `reduction_override` is provided and is not one of [None, 'none', 'mean', 'sum'].

    Note:
        The KL loss is computed as 0.5 * sum(mu^2 + std^2 - 2*std.log() - 1) according to the standard
        formula for the KL divergence between two Gaussian distributions.

    Example:
        >>> mu = torch.tensor([0.0, 1.0, -1.0])
        >>> std = torch.tensor([1.0, 2.0, 0.5])
        >>> loss = kl_loss(mu, std, loss_weight=0.1, reduction_override='mean')
    """
    assert reduction_override in (None, 'none', 'mean', 'sum')

    if reduction_override == 'mean':
        loss = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    else:
        loss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)

    return loss * loss_weight

class KLLoss(nn.Module):
    """
    Kullback-Leibler (KL) Divergence Loss module for Gaussian distributions.

    This module computes the KL divergence loss between a set of Gaussian distributions
    defined by mean (`mu`) and standard deviation (`std`) tensors.

    Args:
        loss_weight (float, optional): Weight for the KL loss. Default is 1.0.
        reduction_override (str, optional): Specifies the reduction method for the loss.
            Should be one of [None, 'mean', 'sum']. Default is None.

    Attributes:
        reduction_override (str): The specified reduction method for the loss.
        loss_weight (float): Weight for the KL loss.

    Methods:
        forward(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            Forward pass of the Kullback-Leibler Divergence Loss.

    Example:
        >>> kl_loss_module = KLLoss(loss_weight=0.5, reduction_override='mean')
        >>> mu = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
        >>> std = torch.tensor([[1.0, 0.5], [0.8, 2.0]])
        >>> loss = kl_loss_module(mu, std)
    """   
    def __init__(self, loss_weight: float = 0.1, reduction_override: str ='mean') -> None:
        super().__init__()
        self.reduction_override = reduction_override
        self.loss_weight = loss_weight

    def forward(self, mu: torch.Tensor, std: torch.Tensor):
        """Forward function of loss.
        """
        losses = []

        for i in range(len(mu)): 
            kl= kl_loss(mu[i], std[i], self.loss_weight, self.reduction_override)
            losses.append(kl)

        return self.loss_weight * sum(losses)/len(mu)
    
if __name__ == "__main__":
    kl_loss_module = KLLoss(loss_weight=0.5, reduction_override='mean')
    shape_in = [
        [1, 512, 28, 28], 
        [1, 1024, 14, 14],
        [1, 2048, 7, 7]
    ]
    mu = []
    std = []
    for i in range(len(shape_in)):
        mu.append(torch.rand(shape_in[i]))
        std.append(torch.rand(shape_in[i]))
    loss = kl_loss_module(mu, std)

    print(loss.item())