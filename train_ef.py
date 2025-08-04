# run_ef.py

import os
import math
import time
import argparse
import numpy as np
import torch
import torchvision
import sklearn.metrics
import matplotlib.pyplot as plt
import tqdm

from echo import *
from model import *  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/n/netscratch/pfister_lab/Everyone/bowen/EchoNet-Dynamic")
    parser.add_argument('--output', type=str, default="./output")
    parser.add_argument('--model_name', type=str, default='r2plus1d_18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=45)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step_period', type=int, default=15)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--period', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("output", f"{args.model_name}_{args.frames}_{args.period}_{'pretrained' if args.pretrained else 'random'}")
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_model = torchvision.models.video.__dict__[args.model_name](pretrained=args.pretrained)
    model = HeteroscedasticEFModel(base_model)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model.to(device)

    if args.weights:
        model.load_state_dict(torch.load(args.weights)['state_dict'])

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_step_period)

    mean_ds, std_ds = get_mean_and_std(Echo(root=args.data_dir, split="train"))
    kwargs = {"target_type": "EF", "mean": mean_ds, "std": std_ds, "length": args.frames, "period": args.period}

    train_loader = torch.utils.data.DataLoader(
        Echo(root=args.data_dir, split="train", pad=12, **kwargs),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    val_loader = torch.utils.data.DataLoader(
        Echo(root=args.data_dir, split="val", **kwargs),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    best_loss = float('inf')
    best_r2 = 0.0

    mean_y, std_y = get_label_mean_and_std(Echo(root=args.data_dir, split="train"))
    print(f"Mean: {mean_y}, Std: {std_y}")

    for epoch in range(args.num_epochs):
        for phase, loader in [('val', val_loader), ('train', train_loader)]:

        # for phase, loader in [('train', train_loader), ('val', val_loader)]:
            model.train(phase == 'train')
            running_loss, preds, targets = 0.0, [], []
            with torch.set_grad_enabled(phase == 'train'):
                for X, y in tqdm.tqdm(loader, desc=f"Epoch {epoch} [{phase}]"):
                    X, y = X.to(device), y.to(device)
                    y = (y - float(mean_y)) / float(std_y)

                    mean, var = model(X)
                    loss = beta_nll_loss(mean, var, y)


                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    running_loss += loss.item() * X.size(0)
                    preds.append(mean.detach().cpu().numpy())
                    targets.append(y.detach().cpu().numpy())

            scheduler.step() if phase == 'train' else None

            preds = np.concatenate(preds)
            targets = np.concatenate(targets)
            r2 = sklearn.metrics.r2_score(targets, preds)
            print(f"Epoch {epoch} [{phase}] Loss: {running_loss/len(loader.dataset):.4f}, R2: {r2:.4f}")

            if phase == 'val' and running_loss < best_loss:
                best_loss = running_loss
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': best_loss,
                }, os.path.join(args.output, 'best_loss.pt'))

            if phase == 'val' and r2 > best_r2:
                best_r2 = r2
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': best_loss,
                }, os.path.join(args.output, 'best_r2.pt'))

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.output, 'latest.pt'))

if __name__ == '__main__':
    main()
