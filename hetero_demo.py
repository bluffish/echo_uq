# hetero_demo.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import random

# Fix random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Create a synthetic dataset: heteroscedastic sine
def make_data(N=500, xmin=0, xmax=10):
    X = np.random.rand(N, 1) * (xmax - xmin) + xmin
    X_sorted = np.sort(X, axis=0)
    y_true = X_sorted * np.sin(X_sorted)    # mean function
    noise_std = 0.3 * X_sorted + 0.1
    y = y_true + noise_std * np.random.randn(N, 1)
    return torch.tensor(X_sorted, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(noise_std**2, dtype=torch.float32)

X, y, y_var = make_data()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Simple MLP that outputs [μ, raw_logvar]
class HeteroMLP(nn.Module):
    def __init__(self, hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.mu = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -2.0)  # softplus ≈ 0.126

    def forward(self, x, detach_for_var=True):
        trunk = self.net(x)
        mu = self.mu(trunk)
        raw = self.logvar(trunk.detach() if detach_for_var else trunk)
        raw = raw.clamp(-10.0, 5.0)
        var = F.softplus(raw) + 1e-6
        var = var.clamp(min=1e-4, max=50.0)
        return mu, var

def beta_nll(mu, var, target, beta=0.5):
    err2 = (mu.squeeze(-1) - target.squeeze(-1)).pow(2)
    logvar = var.log().squeeze(-1)
    pre = 0.5 * (err2 / var.squeeze(-1) + logvar)
    if beta > 0:
        pre = pre * var.detach().pow(beta).squeeze(-1)
    return pre.mean()

def train(model, lr=1e-3, warmup_epochs=10, total_epochs=100, beta=0.5):
    opt = torch.optim.Adam([
        {'params': model.net.parameters(), 'lr': lr},
        {'params': model.mu.parameters(),  'lr': lr},
        {'params': model.logvar.parameters(), 'lr': lr * 0.1},
    ])

    history = {'rmse': [], 'nll': [], 'mean_var': []}

    for epoch in range(total_epochs):
        is_warm = epoch < warmup_epochs
        loss_list = []
        var_means = []
        for Xb, yb in loader:
            mu, var = model(Xb, detach_for_var=not is_warm)
            if is_warm:
                loss = F.mse_loss(mu.squeeze(-1), yb.squeeze(-1))
            else:
                loss = beta_nll(mu, var, yb, beta=beta)
            opt.zero_grad(); loss.backward()
            # nn.utils.clip_grad_norm_(model.mu.parameters(), max_norm=1.0)
            # nn.utils.clip_grad_norm_(model.logvar.parameters(), max_norm=1.0)
            opt.step()

            with torch.no_grad():
                rmse = F.mse_loss(mu.squeeze(-1), yb.squeeze(-1), reduction='mean').sqrt().item()
                nll_single = 0.5 * (torch.log(var) + (yb - mu)**2 / var)
                nll = nll_single.mean().item()
                loss_list.append((rmse, nll))
                var_means.append(var.mean().item())

        rmse_avg = np.mean([r for r, _ in loss_list])
        nll_avg  = np.mean([n for _, n in loss_list])
        history['rmse'].append(rmse_avg)
        history['nll'].append(nll_avg)
        history['mean_var'].append(np.mean(var_means))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            phase = 'Warmup' if is_warm else f'β-NLL (β={beta})'
            print(f"Epoch {epoch+1:>3}/{total_epochs}: {phase}, RMSE={rmse_avg:.4f}, NLL={nll_avg:.4f}, mean σ²={history['mean_var'][-1]:.4f}")

    return history

def visualize(model, X, y, y_var):
    model.eval()
    with torch.no_grad():
        xs = X.squeeze().numpy()
        mu, var = model(X, detach_for_var=False)
        mu = mu.squeeze().numpy()
        sig = np.sqrt(var.squeeze().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, y.squeeze().numpy(), c='k', s=10, label='observed (noisy)')
    plt.plot(xs, mu, 'r-', lw=2, label='predicted μ(x)')
    plt.fill_between(xs, mu - 2 * sig, mu + 2 * sig, color='r', alpha=0.2, label='±2σ')
    # also draw true mean + ±2*true σ
    true_mean = xs * np.sin(xs)
    true_sig = np.sqrt(y_var.squeeze().numpy())
    plt.plot(xs, true_mean, 'C1--', label='true mean')
    plt.fill_between(xs, true_mean - 2 * true_sig, true_mean + 2 * true_sig,
                     color='C1', alpha=0.1, label='true ±2σ')
    plt.legend(); plt.xlabel('x'); plt.ylabel('y'); plt.title('Heteroscedastic β‑NLL demo'); plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = HeteroMLP(hidden=64)
    history = train(model, lr=2e-3, warmup_epochs=0, total_epochs=500, beta=0.5)
    visualize(model, X, y, y_var)
