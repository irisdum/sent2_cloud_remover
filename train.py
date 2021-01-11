# file to train the GAN
from models import clean_gan
from ruamel import yaml
import os
from models import gan_multiGpu

import argparse


def main(path_train, path_model,mgpu):
    if mgpu:
        print("we take care of mgpu")
        gan = gan_multiGpu.GAN(open_yaml(path_model), open_yaml(path_train))
    else:
        gan = clean_gan.GAN(open_yaml(path_model), open_yaml(path_train))
    model_dir = gan.model_dir
    training_dir = gan.this_training_dir
    saving_yaml(path_model, model_dir)
    saving_yaml(path_train, training_dir)
    gan.train()


def saving_yaml(path_yaml, output_dir):
    """

    Args:
        path_yaml: string, path to yaml
        output_dir: string, path to the directory where the yaml is going to be stored

    Returns:

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system("cp {} {}".format(path_yaml, output_dir))


def open_yaml(path_yaml):
    """

    Args:
        path_yaml: string path to the yaml

    Returns:
        
    """
    with open(path_yaml) as f:
        return yaml.load(f)


def _argparser():
    parser = argparse.ArgumentParser(description="Argument GAN train")
    parser.add_argument('--model_path', type=str, default="./GAN_confs/model.yaml",
                        help="path to yaml model ")
    parser.add_argument("--train_path", type=str, default="./GAN_confs/train.yaml")
    parser.add_argument("--mgpu", type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    parser = _argparser()
    # path_train="./GAN_confs/train.yaml"
    # path_model="./GAN_confs/model.yaml"
    main(parser.train_path, parser.model_path,parser.mgpu)
