import torch
import numpy as np


class EFModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        in_features = base_model.fc.in_features
        base_model.fc = torch.nn.Identity()
        self.base_model = base_model

        self.head = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.base_model(x)
        out = self.head(features)
        return out


def mse_loss(mu, y):
    criterion = torch.nn.MSELoss()  
    loss = criterion(mu.unsqueeze(-1), y)
    return loss
