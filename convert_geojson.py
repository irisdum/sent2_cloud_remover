import argparse

from utils.converter import geojson_2_WKT


def main(input_gjson,output_txt):

    geojson_2_WKT(input_gjson,output_txt)

def _argparser():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--input', type=str,default="./confs/train_kangaroo.geojson", help="input path to geojson ")
    parser.add_argument('--output', type=str,default="./confs/train_kangaroo_wkt.txt", help="output path to geojson")

    return parser.parse_args()


if __name__ == '__main__':
    args = _argparser()
    main(args.input,args.output)
