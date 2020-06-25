import os

import buzzard as buzz
import argparse
from utils.image_find_tbx import find_path, extract_tile_id, create_safe_directory
from constant.gee_constant import DICT_ORGA, XDIR,DICT_SHAPE,LABEL_DIR
from utils.load_dataset import load_from_dir


def get_fp(tile_id, build_dataset_dir):
    """:param str the image id
    :returns a buzzard fp"""
    image_path = find_path(build_dataset_dir + XDIR + DICT_ORGA[XDIR][0], tile_id)
    ds = buzz.Dataset(allow_interpolation=True)
    ds.open_raster('tile', image_path)
    return ds.tile.fp


def write_tif_from_fp(array, tile_id, build_dataset_dir, output_dir, prefix=""):
    assert ".tif" in tile_id, "wrong tile id should en with tif"
    ds_tile = buzz.Dataset()
    output_path = "{}{}_image{}".format(output_dir, prefix, tile_id)
    fp_tile = get_fp(tile_id, build_dataset_dir)
    with ds_tile.acreate_raster(output_path, fp_tile, 'float32', channel_count=DICT_SHAPE[LABEL_DIR][-1]).close as cache:
        cache.set_data(array)

    return output_path


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--bd_dir', type=str,
                        help="path to the build_dataset directory ")
    parser.add_argument('--pred_dir', type=str, help="path to the directory where the predictions have been made")
    parser.add_argument('--out_dir', type=str, help="path to the directory where the predictions have been made")
    parser.add_argument('--im_pref', type=str, default="", help="path to the directory where the predictions have been "
                                                                "made")
    parser.add_argument('--path_csv', type=str, default=None, help="Only use to mosaic the gt, the path to the csv gt")
    return parser.parse_args()


def main(build_dataset_dir, predicted_dir, output_dir, im_prefix,path_csv):
    create_safe_directory(output_dir)
    batch_pred, l_path_npy, _ = load_from_dir(predicted_dir, DICT_SHAPE[LABEL_DIR],path_dir_csv=path_csv)
    l_outpath = []
    for i,image_path in enumerate(l_path_npy):
        tile_id = extract_tile_id(image_path).split(".")[0] + ".tif"
        l_outpath += [
            write_tif_from_fp(batch_pred[i, :, :, :], tile_id, build_dataset_dir, output_dir, prefix=im_prefix)]
    os.system("gdal_merge.py {} -o {}".format(" ".join(l_outpath), output_dir + "mosaic.tif"))


if __name__ == '__main__':
    args = _argparser()
    main(args.bd_dir, args.pred_dir, args.out_dir, args.im_pref,args.path_csv)
