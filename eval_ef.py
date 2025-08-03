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
    parser.add_argument('--data_dir', type=str, default="../data/EchoNet-Dynamic")
    parser.add_argument('--output', type=str, default="./test")
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
        model.eval()
        return model

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output, exist_ok=True)

    weight_paths = [os.path.join(args.weights_path, f"{i}/best.pt") for i in range(args.num)]
    models = [load_model(p) for p in weight_paths]
    model = EnsembleModel(models).to(device)

    mean, std = get_mean_and_std(Echo(root=args.data_dir, split="train"))
    kwargs = {"target_type": "EF", "mean": mean, "std": std, "length": args.frames, "period": args.period}

    mean_y, std_y = get_label_mean_and_std(Echo(root=args.data_dir, split="train"))
    print(f"Mean: {mean_y}, Std: {std_y}")

    for split in ["val", "test"]:
        dataset = Echo(root=args.data_dir, split=split, **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device.type == "cuda"))

        all_preds, all_targets, al_vars, ep_vars, abs_errors = [], [], [], [], []

        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for x, y in dataloader:
                    x = x.to(device)
                    b, c, f, h, w = x.shape if len(x.shape) == 5 else x.shape[1:]
                    x = x.view(-1, c, f, h, w)

                    yhat, epistemic_var, var = model(x)
                    yhat = yhat * float(std_y) + float(mean_y)
                    var = var * (float(std_y) ** 2)

                    yhat = yhat.view(-1).cpu().numpy()
                    abs_errors.append(np.abs(yhat.mean() - y.mean()))
                    al_vars.append(var.view(-1).cpu().numpy())
                    ep_vars.append(epistemic_var.view(-1).cpu().numpy())

                    print(yhat)
                    print(y)
                    print("---")

                    all_preds.append(yhat)
                    all_targets.append(y.numpy()[0])
                    pbar.update()

        yhat_mean = np.array([pred.mean() for pred in all_preds])
        y = np.array(all_targets)

        r2 = sklearn.metrics.r2_score(y, yhat_mean)
        mae = sklearn.metrics.mean_absolute_error(y, yhat_mean)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y, yhat_mean))

        print(f"{split} R2: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        with open(os.path.join(args.output, f"{split}_predictions.csv"), "w") as f:
            for fname, preds in zip(dataset.fnames, all_preds):
                for i, p in enumerate(preds):
                    f.write(f"{fname},{i},{p:.4f}\n")

        fig = plt.figure(figsize=(3, 3))
        plt.scatter(y, yhat_mean, color="k", s=1)
        plt.plot([0, 100], [0, 100], linewidth=1, linestyle="--")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"{split}_scatter.pdf"))
        plt.close(fig)

        corr, _ = spearmanr(abs_errors, al_vars)
        print(f"Spearman correlation between |error| and aleatoric uncertainty: {corr:.3f}")


if __name__ == '__main__':
    main()
