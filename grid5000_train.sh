

source /home/idumeur/miniconda3/etc/profile.d/conda.sh
conda activate training_env

export LD_LIBRARY_PATH=/home/idumeur/miniconda3/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/idumeur/miniconda3/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/idumeur/login/miniconda3/include:$CPLUS_INCLUDE_PATH

cp /srv/osirim/idumeur/data/dataset6/prepro1/input_large_dataset.zip  /tmp
unzip tmp/input_large_dataset.zip

python train.py --model_path GAN_confs/model_0.yaml --train_path  GAN_confs/train5.yaml --mgpu true
