# eval_ef.py

import os
import argparse
import numpy as np
import torch
import torchvision
import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics

from echo import *
from model import EFModel

class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        means = []
        for model in self.models:
            mean = model(x)
            means.append(mean)
        means = torch.stack(means, dim=0)
        return means.mean(0), means.var(0)

def fgsm_attack(inputs, targets, model, epsilon, std_y):
    inputs = inputs.clone().detach().requires_grad_(True)
    mean, _, _ = model(inputs)
    mean = mean * float(std_y)  # denormalize
    loss = torch.nn.functional.mse_loss(mean[:, 0], targets)
    loss.backward()
    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = inputs + perturbation
    # adv_inputs = torch.clamp(adv_inputs, 0, 1)  # Keep pixel values valid
    return adv_inputs.detach()

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
    parser.add_argument('--eps', type=float, default=0)
    args = parser.parse_args()

    def load_model(path):
        base = torchvision.models.video.__dict__[args.model_name](pretrained=False)
        model = EFModel(base)
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

        preds, targets, vars, abs_errors = [], [], [], []

        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                if args.eps > 0:
                    x = fgsm_attack(x, y, model, epsilon=args.eps, std_y=std_y)

                mean, var = model(x)
                
                preds.append(mean[:, 0].detach().cpu().numpy())
                targets.append(y.detach().cpu().numpy())
                vars.append(var[:, 0].detach().cpu().numpy())

                abs_errors.append(np.abs(mean[:, 0].detach().cpu().numpy() - y.detach().cpu().numpy()))
                pbar.update()
                
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        vars = np.concatenate(vars)

        abs_errors = np.concatenate(abs_errors)

        r2 = sklearn.metrics.r2_score(targets, preds)
        mae = sklearn.metrics.mean_absolute_error(targets, preds)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(targets, preds))

        print(f"{split} R2: {r2:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"Epistemic variance: {np.mean(vars):.3f}")

        fig = plt.figure(figsize=(3, 3))
        plt.plot([0, 100], [0, 100], linewidth=1, linestyle="--", color="red")
        plt.xlabel("Actual EF (%)")
        plt.ylabel("Predicted EF (%)")
        plt.grid(True)
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"{split}_scatter_uncertainty.pdf"))
        plt.close(fig)


if __name__ == '__main__':
    main()
