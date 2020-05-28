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
    polygon=data["features"][0]
    print(type(polygon))
    print(polygon)
    str_poly = json.dumps(polygon["geometry"])
    g = geojson.loads(str_poly)
    sh=shape(g)
    coordo=sh.bounds
    print("Minx {}, miny {}, maxx {}, maxy{}".format(coordo[0],coordo[1],coordo[2],coordo[3]))
    return "{} {} {} {}".format(coordo[0],coordo[1],coordo[2],coordo[3])


def geojson_2_strcoordo_ul_lr(path_geojson):
    with open(path_geojson) as f:
        data = json.load(f)
    polygon=data["features"][0]
    print(type(polygon))
    print(polygon)
    str_poly = json.dumps(polygon["geometry"])
    g = geojson.loads(str_poly)
    sh=shape(g)
    coordo=sh.bounds
    print("Minx {}, miny {}, maxx {}, maxy{}".format(coordo[0],coordo[1],coordo[2],coordo[3]))
    return "{} {} {} {}".format(coordo[0], coordo[3], coordo[2], coordo[1])

def add_batch_str_coorodo(coordo_str,add_val):
    """:param coordo_str : string
    :returns a string of the coordo where we have added the value of add_val"""
    assert len(add_val)==4, "Wrong in put param add_val should be len 4 not {}".format(add_val)
    l_coordo=coordo_str.split(" ")
    l_increase=[]
    for i,coordo in enumerate(l_coordo):
        l_increase+=[str(float(coordo)+float(add_val[i]))]
    print("We have addee {} to coordo str {} \n The new bbox coordo are {}".format(coordo_str,add_val," ".join(l_increase)))
    return " ".join(l_increase)