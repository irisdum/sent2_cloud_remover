import argparse

from utils.storing_data import create_tiling_hierarchy


def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input_dir', type=str, default="/srv/osirim/idumeur/data/dataset2/prepro3/",
                        help="path to input directory ")
    parser.add_argument('--output_dir', type=str, default="/srv/osirim/idumeur/data/dataset2/prepro3/build_dataset/",
                        help="path to the output directory")
    parser.add_argument("--bands2", nargs="+", default=None, help="list of all the bands of sentinel 2 format B02,B03")

    parser.add_argument("--bands1", nargs="+", default=None, help="list of all the bands of sentinel1 format vv, vh")
    parser.add_argument("--geojson", default="./confs/train_kangaroo_utm2.geojson", help="path to the zone geojson")
    parser.add_argument("--overlap",type=int, default=0, help="path to the zone geojson")
    return parser.parse_args()

def main(input_dir, output_dir, list_band2, list_band1, path_geojson):
    """

    Args:
        input_dir:
        output_dir:
        list_band2:
        list_band1:
        path_geojson:

    Returns:

    """
    create_tiling_hierarchy(output_dir)
    # First if numerous S1 images : merge them

    # Mosaic each image with the bands and create a vrt
    # Crop the image
    # Tile the image
    pass
def process_date_sent(list_band, sent, input_dir, output_dir, sub_dir, t, path_geojson):
    """

    Args:
        list_band: list of string, indicates the band which are going to be used
        sent: int, could be 1 or 2, respectively for sentinel 1 data or sentinel 2 data
        input_dir: string path to the directory which contains the preprocesssed image
        output_dir: string The name of the directory for the dataset
        sub_dir: directory name for label of x data. (For the training)
        t: index used in the DIR_T, gives the which temporal directory should be used
        path_geojson:

    Returns:

    """

if __name__ == '__main__':
    args = _argparser()
    main(args.input_dir, args.output_dir, args.bands2, args.bands1, args.geojson)
