# eval_ef.py

import os
import argparse
import numpy as np
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.stats import spearmanr

from echo import *
from model import HeteroscedasticEFModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/n/netscratch/pfister_lab/Everyone/bowen/EchoNet-Dynamic")    
    parser.add_argument('--output', type=str, default="./output")

    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='r2plus1d_18')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--period', type=int, default=2)
    parser.add_argument('--num', type=int, default=1)
    args = parser.parse_args()

    class EnsembleModel(torch.nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = torch.nn.ModuleList(models)

        def forward(self, x):
            means, log_vars = [], []
            for model in self.models:
                mean, log_var = model(x)
                means.append(mean)
                log_vars.append(log_var)
            means = torch.stack(means, dim=0)
            log_vars = torch.stack(log_vars, dim=0)
            return means.mean(0), means.var(0), log_vars.mean(0)

    def load_model(path):
        base = torchvision.models.video.__dict__[args.model_name](pretrained=False)
        model = HeteroscedasticEFModel(base)
        model = torch.nn.DataParallel(model).to(device)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        print(checkpoint['epoch'])
        model.eval()
        return model

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output, exist_ok=True)

    weight_paths = [os.path.join(args.weights_path, f"{i}/best_r2.pt") for i in range(args.num)]
    models = [load_model(p) for p in weight_paths]
    model = EnsembleModel(models).to(device)

    mean, std = get_mean_and_std(Echo(root=args.data_dir, split="train"))
    kwargs = {"target_type": "EF", "mean": mean, "std": std, "length": args.frames, "period": args.period}

    mean_y, std_y = get_label_mean_and_std(Echo(root=args.data_dir, split="train"))
    print(f"Mean: {mean_y}, Std: {std_y}")

    for split in ["val", "test"]:
        dataset = Echo(root=args.data_dir, split=split, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

        preds, targets, al_vars, ep_vars, abs_errors = [], [], [], [], []

        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    #y = (y - float(mean_y)) / float(std_y)

                    mean, ep_var, var = model(x)
                    
                    mean = mean * float(std_y) + float(mean_y)
                    var = var * float(std_y) ** 2

                    print(var)

                    preds.append(mean[:, 0].detach().cpu().numpy())
                    targets.append(y.detach().cpu().numpy())
                    al_vars.append(var[:, 0].detach().cpu().numpy())
                    ep_vars.append(ep_var.detach().cpu().numpy())

                    pbar.update()
                    
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        al_vars = np.concatenate(al_vars)

        r2 = sklearn.metrics.r2_score(targets, preds)
        mae = sklearn.metrics.mean_absolute_error(targets, preds)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(targets, preds))

        print(f"{split} R2: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

#        with open(os.path.join(args.output, f"{split}_predictions.csv"), "w") as f:
#            for fname, preds in zip(dataset.fnames, preds):
#                for i, p in enumerate(preds):
#                    f.write(f"{fname},{i},{p:.4f}\n")
#
        yhat_std = np.array([np.sqrt(v) for v in al_vars])  # mean aleatoric std per sample
        print(targets.shape, preds.shape, yhat_std.shape)
        fig = plt.figure(figsize=(3, 3))
        plt.errorbar(targets, preds, yerr=2 * yhat_std, fmt='o', markersize=2, ecolor='gray', alpha=0.5, capsize=2, label='±2σ')
        plt.plot([0, 100], [0, 100], linewidth=1, linestyle="--", color="red")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.grid(True)
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"{split}_scatter_uncertainty.pdf"))
        plt.close(fig)

        spearman_corr, spearman_p = spearmanr(yhat_std, abs_errors)
        print(f"{split} Spearman correlation between uncertainty and absolute error: r = {spearman_corr:.3f}, p = {spearman_p:.3g}")

        # Optional: Pearson correlation as well
        pearson_corr = np.corrcoef(yhat_std, abs_errors)[0, 1]
        print(f"{split} Pearson correlation between uncertainty and absolute error: r = {pearson_corr:.3f}")


if __name__ == '__main__':
    main()
