#file to train the GAN
from models import GAN
from ruamel import yaml

def main(path_train,path_model):

    model=GAN.GAN(open_yaml(path_train),open_yaml(path_model))

def open_yaml(path_yaml):
    with open(path_yaml) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

if __name__ == '__main__':
    path_train="./GAN_confs/train.yaml"
    path_model="./GAN_confs/train.yaml"
    main(path_train,path_model)