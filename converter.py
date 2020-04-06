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

def main():
    geojson_2_WKT("./confs/grid_train_kangaroo.geojson","./confs/grid_train_kangaroo_wkt.txt")


if __name__ == '__main__':
    main()
