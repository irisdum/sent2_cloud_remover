#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --qos=express
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128g

# first parameter is a path to the graph xml
model_path="$1"

# second parameter is a path to a parameter file
train_path="$2"

python train.py --model_path ${model_path} --train_path ${train_path}