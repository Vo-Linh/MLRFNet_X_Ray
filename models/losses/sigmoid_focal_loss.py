import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2,
                 reduction='mean',
                 loss_weight=1.0) -> None:
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
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
        >>> sfl_loss = SigmoidFocalLoss(beta=1.0, 
                                       alpha=0.4, s=0.2, 
                                       reduction='mean', 
                                       loss_weight=1.0)
        >>> loss = sfl_loss(pred, target)
        >>> assert loss.shape == torch.Size([])
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * sigmoid_focal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            **kwargs)
        return loss


if __name__ == "__main__":
    pred = torch.randn(1, 100)
    target = torch.randint(0, 100, (1, 100))
    bfl_loss = SigmoidFocalLoss(beta=1.0,
                               alpha=0.4, s=0.2,
                               reduction='mean',
                               loss_weight=1.0)
    loss = bfl_loss(pred, target)
    print(loss)
