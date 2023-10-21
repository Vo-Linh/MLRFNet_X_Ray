import torch
from torch import nn


def biased_focal_loss(pred, target, beta=1.0, alpha=0.4, s=0.2, reduction='mean'):
    """
    Args:
        pred (torch.Tensor): The prediction with shape (B, N).
        target (torch.Tensor): The learning target of the prediction with
            shape (B, N).
        beta (float): The loss is a piecewise function of prediction and target
            and ``beta`` serves as a threshold for the difference between the
            prediction and target. Defaults to 1.0.
        alpha (float): The denominator ``alpha`` in the focal loss.
            Defaults to 0.4.
        s (float): probability shift factor s is set to make the network focus 
                    more on the hard-to-classify parts of negative samples.
                    Defaults to 0.2.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".

    Returns:
        torch.Tensor: The calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()

    loss = -alpha * (1 - pred)**beta * target * torch.log(pred) - \
        (1 - alpha) * (pred**s)**beta * (1 - target) * torch.log(1 - pred)

    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()

    return loss


class BiasedFocalLoss(nn.Module):
    def __init__(self,
                 beta=1.0,
                 alpha=0.4,
                 s=0.2,
                 reduction='mean',
                 loss_weight=1.0) -> None:
        super(BiasedFocalLoss, self).__init__()
        self.alpha = alpha
        self.s = s
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                reduction_override=None,
                **kwargs):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction with shape (B, N).
            target (torch.Tensor): The learning target of the prediction with
                shape (B, N).
            weight (torch.Tensor, optional): Sample-wise loss weight with
                shape (N, ).
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        >>> pred = torch.randn(1, 100)
        >>> target = torch.randint(0, 100, (1, 100))
        >>> bfl_loss = BiasedFocalLoss(beta=1.0, 
                                       alpha=0.4, s=0.2, 
                                       reduction='mean', 
                                       loss_weight=1.0)
        >>> loss = bfl_loss(pred, target)
        >>> assert loss.shape == torch.Size([])
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * biased_focal_loss(
            pred,
            target,
            beta=self.beta,
            alpha=self.alpha,
            s=self.s,
            reduction=reduction,
            **kwargs)
        return loss


if __name__ == "__main__":
    pred = torch.randn(1, 100)
    target = torch.randint(0, 100, (1, 100))
    bfl_loss = BiasedFocalLoss(beta=1.0,
                               alpha=0.4, s=0.2,
                               reduction='mean',
                               loss_weight=1.0)
    loss = bfl_loss(pred, target)
    print(loss)
