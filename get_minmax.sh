#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --qos=express
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128g

build_dataset_dir="$1"
path_input_data="$2"
begin_date="$3"
end_date="$4"
vi="$5"

python gee_ndvi_minmax.py --path_bdata ${build_dataset_dir} --path_input_data ${path_input_data} --bd ${begin_date} --ed ${end_date} --vi ${vi}


