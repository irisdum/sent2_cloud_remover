#file to train the GAN
from models import GAN
from ruamel import yaml
import tensorflow as tf

def main(path_train,path_model):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        gan=GAN.GAN(open_yaml(path_train),open_yaml(path_model),sess)
        gan.build_model()
        gan.train()
        
def open_yaml(path_yaml):
    with open(path_yaml) as f:
        return yaml.load(f)

if __name__ == '__main__':
    path_train="./GAN_confs/train.yaml"
    path_model="./GAN_confs/model.yaml"
    main(path_train,path_model)