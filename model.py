import torch
import numpy as np

class HeteroscedasticEFModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        in_features = base_model.fc.in_features
        base_model.fc = torch.nn.Identity()
        self.base_model = base_model

        self.head = torch.nn.Linear(in_features, 1)
        self.logvar = torch.nn.Linear(in_features, 1)

        # self.head.bias.data.fill_(55.6)
        # self.logvar.bias.data.fill_(float(np.log(1.0 - 1e-6)))

    def forward(self, x):
        features = self.base_model(x)
        mean = self.head(features)
        logvar = self.logvar(features).clamp(-10, 5)        
        var = torch.exp(logvar) + 1e-6

        return mean, var


def heteroscedastic_loss(mu, var, y):
    # var = torch.exp(log_var)
    # loss = 0.5 * log_var + 0.5 * ((y - mu)**2 / var)

    # return loss.mean()
    # criterion = torch.nn.GaussianNLLLoss()  
    # loss = criterion(mu, y, var)
    # return loss

    criterion = torch.nn.MSELoss()  
    loss = criterion(mu.unsqueeze(-1), y)
    return loss

def beta_nll_loss(mu, var, target, beta=.5, reduction="mean"):
    """
    Args:
        mu: Tensor (B,) or (B,1) - predicted mean
        var: Tensor (B,) - predicted positive variance
        target: Tensor (B,) - ground-truth
        beta: float in [0,1]
        reduction: "mean" or "sum" or "none"
    Returns:
        scalar (if reduction="mean" or "sum") or Tensor (B,)
    """        
    loss_elem = 0.5 * ((target - mu.squeeze(-1)).pow(2) / var + var.log()) 

    if beta > 0:
        loss_elem = loss_elem * var.detach().pow(beta)

    if reduction == "mean":
        return loss_elem.mean()
    elif reduction == "sum":
        return loss_elem.sum()
    else:
        return loss_elem