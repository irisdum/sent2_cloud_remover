import numpy as np
#File for the storing constant, names of the files and directory used to download the images

DIR_SENT=["sentinel1/", "sentinel2/"]
DOWNLOAD_PATH= "./Downloads/" # /datastore/dum031/data/dataset1/ #path where the image are download from sara
TEMPORARY_DIR= "temporary_dir/"
TILING_DIR="tiling_dir/"
XDIR="dataX/"
LABEL_DIR="label/"
DIR_T=["date1/", "date2/","date3/"]
OPT_DWND_IMAGE="zip" #the format of the file to dwnld avoid changing

LIST_XDIR=["Sentinel1_t{}".format(i) for i in range(len(DIR_T))]+["Sentinel2_t{}".format(i) for i in range(len(DIR_T)-1)]
LIST_LABEL_DIR=["Sentinel2_t{}".format(len(DIR_T)-1)]
DICT_ORGA={XDIR:LIST_XDIR, LABEL_DIR:LIST_LABEL_DIR} #Should be modified if multiple date
DICT_ORGA_INT={XDIR:[(1,t) for t in range(len(DIR_T))]+[(2,t)for t in range(len(DIR_T)-1) ],LABEL_DIR:[(2,len(DIR_T)-1)]}
URL_FILE="downloaded_im_url.txt"

DICT_SHAPE={XDIR:(256, 256, 8), LABEL_DIR:(256, 256, 4)}