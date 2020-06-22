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



module load miniconda3/4.3.13
unset PYTHONHOME
unset PYTHONPATH
source activate /datastore/dum031/envs/env_tf_gpu
python predict.py --model_path ${model_path}   --tr_nber ${train_nber} --weights ${weight}

