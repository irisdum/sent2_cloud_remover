#file to train the GAN
from models import GAN
from ruamel import yaml
import tensorflow as tf

from processing import create_safe_directory


def main(path_train,path_model):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)sess = tf.Session(config=config)
    with tf.compat.v1.Session(config=config) as sess:
        train_yaml=open_yaml(path_train)
        create_safe_directory(train_yaml["result_dir"])

        gan=GAN.GAN(train_yaml,open_yaml(path_model),sess)
        gan.build_model()
        gan.train()

def open_yaml(path_yaml):
    with open(path_yaml) as f:
        return yaml.load(f)

if __name__ == '__main__':
    path_train="./GAN_confs/train.yaml"
    path_model="./GAN_confs/model.yaml"
    main(path_train,path_model)