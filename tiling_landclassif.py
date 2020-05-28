import argparse
import os

from constant.gee_constant import EPSG_LANDCLASS, EPSG
from processing import crop_image, tiling,  create_safe_directory


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
    os.system("gdalwarp  -t_srs {} -tr 10 10 {} {}".format(EPSG,path_tif,path_tif.split(".")[0]+"reproj.tiff"))
    crop_image_name = crop_image(path_tif.split(".")[0]+"reproj.tiff", path_geojson,
                                 output_dir + "crop_aus18.vrt")
    os.system("gdalinfo {}".format(crop_image_name))
    shp_file_t1 = tiling(crop_image_name, output_dir, 4, 0)

if __name__ == '__main__':
    args = _argparser()
    main(args.path_tif,args.output_dir,args.geojson)