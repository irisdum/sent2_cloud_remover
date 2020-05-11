#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --qos=express
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128g

python train.py --model_path ./GAN_confs/model.yaml --train_path ./GAN_confs/train.yaml