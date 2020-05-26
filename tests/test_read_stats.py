#This file enables us to rain numerous test
import sys
sys.path.append("..")
import argparse
from scanning_dataset import extract_tile_id
from constant.gee_constant import LABEL_DIR, DICT_SHAPE
from utils.display_image import find_image_indir
from utils.load_dataset import load_from_dir
from utils.normalize import get_minmax_fromcsv, find_csv


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
        tile_id=extract_tile_id(find_image_indir(input_dataset+LABEL_DIR, "npy")[0])
        path_csv=find_csv(dataset_dir,"B2")
        val_min, val_max=get_minmax_fromcsv(tile_id.split(".")[0] + ".tif", path_csv, "B2")
        print("TEST for image {} the min_max from csv is {}".format(tile_id, (val_min, val_max)))

        print("load_from_dir function")
        data_array,path_tile,ldict_stat=load_from_dir(input_dataset+LABEL_DIR, DICT_SHAPE[LABEL_DIR], path_dir_csv=dataset_dir)
        assert ldict_stat is not None, "Wrong output should be a list"
        assert type(ldict_stat)==type([]),"The ouput of the function should be a list not {}".format(type(ldict_stat))
        assert data_array.shape[0]==len(ldict_stat),"The batch size and the len of ldict_stat dos not match {}".format(len(ldict_stat))
        print(ldict_stat)
        assert type(ldict_stat[0])==type({}),"Inside the list should be dict not {}".format(ldict_stat[0])


if __name__ == '__main__':
    args=_argparser()
    main(args.test_name,args.dataset_dir,args.input_dataset)