#This file enables us to rain numerous test
import sys
sys.path.append("..")
import argparse
import numpy as np
from utils.image_find_tbx import extract_tile_id, find_image_indir, find_csv
from constant.gee_constant import DICT_SHAPE
from constant.storing_constant import XDIR, LABEL_DIR
from utils.load_dataset import load_from_dir, load_data
from utils.normalize import get_minmax_fromcsv


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--dataset_dir', type=str, default="/datastore/dum031/data/dataset2/",
                        help="path to input  build_dataset directory ")
    parser.add_argument('--test_name',type=str,default="read_csv_stats",help="name of the test to run")
    parser.add_argument('--input_dataset', type=str, default="/datastore/dum031/data/dataset2/prepro1/input_dataset/val/", help="name of the test to run")
    return parser.parse_args()


def main(test_name,dataset_dir,input_dataset):
    print("[INFO] test {} is going to be runned".format(test_name))
    if test_name=="read_csv_stats":
        tile_id=extract_tile_id(find_image_indir(input_dataset+XDIR, "npy")[0])
        path_csv=find_csv(dataset_dir,"B2")
        val_min, val_max=get_minmax_fromcsv(tile_id.split(".")[0] + ".tif", path_csv, "B2")
        print("TEST for image {} the min_max from csv is {}".format(tile_id, (val_min, val_max)))

        print("load_from_dir function")
        data_array,path_tile,ldict_stat=load_from_dir(input_dataset+XDIR, DICT_SHAPE[XDIR], path_dir_csv=dataset_dir)
        assert ldict_stat is not None, "Wrong output should be a list"
        assert type(ldict_stat)==type([]),"The ouput of the function should be a list not {}".format(type(ldict_stat))
        assert data_array.shape[0]==len(ldict_stat),"The batch size and the len of ldict_stat dos not match {}".format(len(ldict_stat))
        print(ldict_stat)
        assert type(ldict_stat[0])==type({}),"Inside the list should be dict not {}".format(ldict_stat[0])
        print("[TEST] load_data function")
        dataX, data_label= load_data(input_dataset, x_shape=None, label_shape=None, normalization=True,
                                     dict_band_X=None, dict_band_label=None, dict_rescale_type=None)
        print("Using the csv stats for s2 and normalize",np.mean(dataX[0,:,:,4]),np.mean(data_label[0,:,:,0]))
        dataX, data_label = load_data(input_dataset, x_shape=None, label_shape=None, normalization=True,
                                      dict_band_X=None, dict_band_label=None, dict_rescale_type=None)
        print("Using the previous normalization method",np.mean(dataX[0, :, :, 4]), np.mean(data_label[0, :, :, 0]))

if __name__ == '__main__':
    args=_argparser()
    main(args.test_name,args.dataset_dir,args.input_dataset)