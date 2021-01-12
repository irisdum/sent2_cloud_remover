# first parameter is a path to the graph xml
model_path="$1"

# second parameter is a path to a parameter file
train_path="$2"

dataset_path="$3"

source /home/idumeur/miniconda3/etc/profile.d/conda.sh
conda activate training_env



cp ${dataset_path} -R /tmp

python train.py --model_path ${model_path} --train_path ${train_path} --mgpu true
