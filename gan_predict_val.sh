#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --qos=express
#SBATCH --time=0:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128g

# first parameter is a path to the graph xml
model_path="$1"

# second parameter is a path to a parameter file
train_nber="$2"

weight="$3"



source /home/idumeur/miniconda3/etc/profile.d/conda.sh
conda activate training_env

python predict.py --model_path ${model_path}   --tr_nber ${train_nber} --weights ${weight}

