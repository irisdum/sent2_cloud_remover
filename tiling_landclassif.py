import argparse
import glob
import os
import numpy as np
from constant.gee_constant import EPSG_LANDCLASS, EPSG
from processing import crop_image, tiling,  create_safe_directory
from utils.converter import geojson_2_strcoordo_ul_lr, add_batch_str_coorodo


def get_landclass_tile(path_landclass_dir,tile_id):
    """:param"""
    l=glob.glob('{}**/*{}*'.format(path_landclass_dir,tile_id.split(".")[0]),recursive=True)
    assert len(l)==1,"No image found at {} for tile id {} \n using glob command {} ".format(path_landclass_dir,tile_id.split(".")[0],'{}**/*{}*'.format(path_landclass_dir,tile_id.split(".")[0]))
    return l[0]

def load_landclass_tile(path_landclass_dir,tile_id):
    path_tile=get_landclass_tile(path_landclass_dir,tile_id)
    return np.load(path_tile)


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--path_tif', type=str, default="/datastore/dum031/data/dataset2/aus_for18.tiff",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/datastore/dum031/data/dataset2/build_dataset_land_classif",
                        help="path to the output directory")

    parser.add_argument("--geojson", default="./confs/train_kangaroo_utm2.geojson", help="path to the zone geojson")

    return parser.parse_args()




def main(path_tif,output_dir,path_geojson):
    create_safe_directory(output_dir)
    os.system("gdalwarp  -t_srs {} {} {}".format(EPSG,path_tif,path_tif.split(".")[0]+"_reproj.tiff"))
    crop_image_name = output_dir + "crop_aus18.vrt"
    str_bbox = geojson_2_strcoordo_ul_lr(path_geojson)
    str_bbox_increase=add_batch_str_coorodo(str_bbox,[1000,1000,1000,1000]) #TODO add it in the constant file
    #first resample using a small batch area, then once the resolution is set at 10 resample exactly
    os.system(
        "gdal_translate {} {} -projwin  {} -projwin_srs {} -tr 10 10".format(path_tif.split(".")[0] + "_reproj.tiff",
                                                                             path_tif.split(".")[0] + "_reproj2.tiff", str_bbox_increase, EPSG))
    os.system(
        "gdal_translate {} {} -projwin  {} -projwin_srs {} -tr 10 10".format(path_tif.split(".")[0]+"_reproj2.tiff",crop_image_name, str_bbox, EPSG))
    os.system("gdalinfo {}".format(crop_image_name))
    shp_file_t1 = tiling(crop_image_name, output_dir, 4, 0)

if __name__ == '__main__':
    args = _argparser()
    main(args.path_tif,args.output_dir,args.geojson)