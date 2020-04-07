import argparse

import geojson
from shapely.geometry import shape
import json

def geojson_2_WKT(path_geojson,path_wkt):
    with open(path_geojson) as f:
        data = json.load(f)
    f=open(path_wkt, "w")
    for polygon in data["features"]:
        str_poly=json.dumps(polygon["geometry"])
        #print(str_poly)
        #print(type(str_poly))
        g=geojson.loads(str_poly)
        #shp=shape(g)
        wkt_str=shape(g).wkt
        f.write(wkt_str+"\n")
        print(wkt_str)
    f.close()

def geojson_2_bboxcoordo(path_geojson):
    with open(path_geojson) as f:
        data = json.load(f)
    polygon=data["features"]
    str_poly = json.dumps(polygon["geometry"])
    g = geojson.loads(str_poly)
    sh=shape(g)
    coordo=sh.bounds
    print("Minx {}, miny {}, maxx {}, maxy{}".format(coordo[0],coordo[1],coordo[2],coordo[3]))
    return "{} {} {} {}".format(coordo[0],coordo[1],coordo[2],coordo[3])

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
