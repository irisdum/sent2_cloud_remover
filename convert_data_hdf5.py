import argparse
import numpy as np
import h5py
import os
from constant.model_constant import TRAINING_DIR
from train import open_yaml
from utils.load_dataset import load_data


def convert_all_data(input_dir, path_model, path_train):
    """

    Args:
        input_dir:
        train_yaml:

    Returns:

    """
    model_yaml = open_yaml(path_model)
    train_yaml = open_yaml(path_train)
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


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str,
                        default="/srv/osirim/idumeur/data/dataset6/prepro1/input_large_dataset",
                        help="path to input  build_dataset directory ")
    parser.add_argument('--train_path', type=str,
                        default="/GAN_confs/train_vf.yaml",
                        help="path to  train yaml")
    parser.add_argument('--model_path', type=str,
                        default="/GAN_confs/model.yaml",
                        help="path to model yaml ")

    return parser.parse_args()


if __name__ == '__main__':
    args = _argparser()
    convert_all_data(args.input_dir,args.train_path,args.model_path)
