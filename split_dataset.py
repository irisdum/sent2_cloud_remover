import numpy as np
from scanning_dataset import list_all_conformed_tiles,split_train_test_val
from utils.create_dataset import create_input_dataset
import argparse

def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--ptrain',type=float,default=0.80,help="The percentage of image to put into the train dataset")
    parser.add_argument('--pval', type=float,default=0.05, help="The percentage of image to put into the val dataset")
    parser.add_argument('--ptest', type=float,default=0.15, help="The percentage of image to put into the test dataset")
    parser.add_argument("--input_dataset",type=str,default="/datastore/dum031/data/dataset2/prepro1/build_dataset/",help="Path to the build_dataset")
    parser.add_argument("--output_dir_name",type=str,default="input_large_dataset")
    parser.add_argument("--random_state",type=int,default=2)
    parser.add_argument("--norm", default=False, help="Bool if set to true apply a normalization while creating the dataset")
    parser.add_argument("--keep_clouds", default=False,
                        help="Bool if set to true the tile with clouds are not removed from the dataset")
    parser.add_argument("--tile_size", default=256,
                        help="tile size")
    return parser.parse_args()




def main(input_dir,output_dir_name,ptrain,pval,ptest,rd_state,keep_clouds,tile_size):
    path_output_dir="/".join(input_dir.split("/")[:-2]+[output_dir_name,""])
    l_tiles_tot = list_all_conformed_tiles(input_dir,keep_clouds=keep_clouds)
    dict_data_selected = split_train_test_val(l_tiles_tot, ptrain,pval,ptest, random_state=rd_state)
    create_input_dataset(dict_data_selected,input_dir, path_output_dir,tile_size=tile_size)

if __name__ == '__main__':
    args=_argparser()
    main(args.input_dataset,args.output_dir_name,args.ptrain,args.pval,args.ptest,args.random_state,args.keep_clouds,
         args.tile_size)

