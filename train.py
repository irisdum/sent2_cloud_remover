# file to train the GAN
from constant.model_constant import TRAINING_DIR
from models import clean_gan
from ruamel import yaml
import os
from models import gan_multiGpu
import h5py
import argparse

from utils.load_dataset import load_data


def main(path_train, path_model,mgpu,h5):
    train_yaml=open_yaml(path_train)
    model_yaml=open_yaml(path_model)
    input_dir="/".join(train_yaml["train_directory"].split("/")[:-2])+"/"
    if h5:
        dict_path_data_h5py=convert_all_data(input_dir,train_yaml,model_yaml=model_yaml)
    else:
        dict_path_data_h5py=None
    print("ALl data in h5py format in {}".format(dict_path_data_h5py))

    gan = clean_gan.GAN(model_yaml, train_yaml,data_h5py=dict_path_data_h5py)
    model_dir = gan.model_dir
    training_dir = gan.this_training_dir
    saving_yaml(path_model, model_dir)
    saving_yaml(path_train, training_dir)
    gan.train()

def convert_all_data(input_dir, train_yaml, model_yaml):
    """

    Args:
        input_dir:
        train_yaml:

    Returns:

    """
    list_final_path=[]
    for subdir in TRAINING_DIR:
        path_final_data = input_dir + subdir[:-1] + "dataset.h5"
        if not os.path.isfile(path_final_data):
            create_h5_data(input_dir + subdir, path_final_data, train_yaml, model_yaml)
        else:
            print("File {} already exists ".format(path_final_data))
        list_final_path+=[path_final_data]
    return dict(zip(TRAINING_DIR,list_final_path))

def create_h5_data(input_dir, output_path, train_yaml, model_yaml):
    """
    #TODO be careful if you change of normalization
    Args:
        input_dir: For all npy tile in a directory, save them with the normalization parameter into a .hpy5 format
        output_path: For all npy tile in a directory, save them with the normalization parameter into a .hpy5 format
        train_yaml: For all npy tile in a directory, save them with the normalization parameter into a .hpy5 format
        model_yaml: For all npy tile in a directory, save them with the normalization parameter into a .hpy5 format

    Returns:

    """
    data_X, data_y, scale_dict_train = load_data(input_dir,
                                                 x_shape=model_yaml["input_shape"],
                                                 label_shape=model_yaml["dim_gt_image"],
                                                 normalization=train_yaml["normalization"],
                                                 dict_band_X=train_yaml["dict_band_x"],
                                                 dict_band_label=train_yaml["dict_band_label"],
                                                 dict_rescale_type=train_yaml["dict_rescale_type"],
                                                 fact_s2=train_yaml["s2_scale"], fact_s1=train_yaml["s1_scale"],
                                                 s2_bands=train_yaml["s2bands"], s1_bands=train_yaml["s1bands"],
                                                 lim=train_yaml["lim_train_tile"])

    hf = h5py.File(output_path, 'w')
    hf.create_dataset('data_X', data=data_X)
    hf.create_dataset('data_y', data=data_y)
    hf.close()
    return output_path


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
    parser.add_argument("--h5", type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    parser = _argparser()
    # path_train="./GAN_confs/train.yaml"
    # path_model="./GAN_confs/model.yaml"
    main(parser.train_path, parser.model_path,parser.mgpu,parser.h5)
