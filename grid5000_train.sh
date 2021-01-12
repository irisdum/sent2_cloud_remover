#!/bin/bash
# first parameter is a path to the graph xml


source /home/idumeur/miniconda3/etc/profile.d/conda.sh
conda activate training_env



cp /srv/osirim/idumeur/data/dataset6/prepro1/input_large_dataset.zip  /tmp
unzip tmp/input_large_dataset.zip

python train.py --model_path GAN_confs/model_0.yaml --train_path  GAN_confs/train_28.yaml 
