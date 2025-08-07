#!/bin/bash
#SBATCH -c 8
#SBATCH -t 15:00:00
#SBATCH -p seas_gpu
#SBATCH --mem=8000
#SBATCH --gres=gpu:1
#SBATCH --output=./output/video-%j.out

source ~/.bashrc

conda activate dynamic
python train_ef.py --seed 0 --output output/0
python train_ef.py --seed 1 --output output/1
python train_ef.py --seed 2 --output output/2
python train_ef.py --seed 3 --output output/3
python train_ef.py --seed 4 --output output/4
python train_ef.py --seed 5 --output output/5
python train_ef.py --seed 6 --output output/6
python train_ef.py --seed 7 --output output/7
python train_ef.py --seed 8 --output output/8
python train_ef.py --seed 9 --output output/9

