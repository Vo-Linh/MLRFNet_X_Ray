import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
def feature_selection_loss(pred: torch.Tensor,
                           gamma: float = 1.0) -> torch.Tensor:
    """Calculates the feature selection loss.

    Args:
        pred (torch.Tensor): The predicted probability tensor.
        gamma (float, optional): The balance coefficient between entropy divergence and mean. Defaults to 1.0.

    Returns:
        torch.Tensor: The feature selection loss.
    """
    entropy_dis = Bernoulli(pred).entropy().mean()
    mean = torch.mean(pred)

    return entropy_dis, gamma * mean


class FeatureSelectionLoss(nn.Module):
              
    def __init__(self, gamma: float = 1.0, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction with shape (B, N).

        Returns:
            torch.Tensor: The calculated loss
        >>> pred = torch.randn(1, 100)
        >>> fs_loss = FeatureSelectionLoss(gammma = 1.0
                                            loss_weight=1.0)
        >>> loss = fs_loss(pred)
        """
        loss = []
        mean_fs = []
        for i in range(len(pred)): 
            entropy_dis, mean = feature_selection_loss(pred[i])
            loss.append(entropy_dis)
            mean_fs.append(mean)

        return self.loss_weight * sum(loss)/len(pred), sum(mean_fs)/len(pred)

if __name__ == "__main__":
    pred = torch.sigmoid(torch.randn(1, 1, 28, 28))
    fs_loss = FeatureSelectionLoss(gamma= 1.0,
                                    loss_weight=1.0)
    loss = fs_loss(pred)
    print(loss.item())