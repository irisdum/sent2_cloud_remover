#CONSTANT TO FILTER RESULTS TO FIND AND DOWNLOAD IMAGES
from constant.storing_constant import XDIR, LABEL_DIR

DICT_COLLECTION={1: "COPERNICUS/S1_GRD", 2: "COPERNICUS/S2_SR"} # corresponds to the gee collection name
S2_OPTPARAM={"instrument": "MSI", "processingLevel": "L2A"}
S1_OPTPARAM={"productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR"} # "polarisation": "VH" "orbitDirection":"descending"
ORBIT_ID={1: "orbitNumber_start", 2: "MGRS_TILE"}

# STORING IMAGES CONSTANT
SENT_FORMAT=["tiff", "jp2"]
LISTE_BANDE=[["VV","VH"],["B04", "B03", "B02","B08"]] #for downloading the data avoid changing its value
GEE_S2_BAND=["B2","B3","B4","B8"]
    #[["vh","vv"],["B04", "B03", "B02","B08","cm"]]



### DOWNLOADING CRITERIA
FACTEUR_AREA=0.99 # correspond of the minimum area of the zone the sentinel 1 should cover

VAR_NAME="_prepro"

## TILING OPTION
OVERLAP=10  #overlap of the tiling

##PROJECTION SENTINEL 2
EPSG="EPSG:32755"
#Land calssif EPSG
EPSG_LANDCLASS="EPSG:3577"

## DISPALY CONSTANT
BOUND_X=[100,1000]
BOUND_Y=[100,1000]

## CONVERT Uint16 2 Float 32
CONVERTOR=10000 #apply when displaying the tif tile but also when creating the train, test, val dataset
SCALE_S1=1

## SCAN TILES THRESHOLD OF NUMBER OF CLOUD PIXEL ALLOWED TO KEEP THE TILE
CLOUD_THR=20
TOT_ZERO_PIXEL=0.7 #the minimum percentage of non 0 pixels that the tile should have to avoid being removed

## Dataset tiles shape
DICT_SHAPE={XDIR:(256, 256, 8), LABEL_DIR:(256, 256, 4)}

##The data rescaling before going into the NN
DICT_BAND_LABEL={"R":[0],"G":[1],"B":[2],"NIR":[3]}
DICT_BAND_X={"VV":[0,2],"VH":[1,3],"R":[4],"G":[5],"B":[6],"NIR":[7]}
DICT_RESCALE_REVERSE={"R":"center_norm11_r","G":"center_norm11_r","B":"center_norm11_r","NIR":"center_norm11_r","VV":"centering_r",
              "VH":"centering_r"} #TODO remove

DICT_METHOD={"standardization": "mean_std","standardization11": "mean_std", "centering": "mean_std",
             "normalization": "min_max"," ": "min_max","centering_r":"mean_std","normalization11_r":"min_max",
             "center_norm11":"min_max","center_norm11_r":"min_max"}
#TODO remove think is unused

#TRAINING CONSTANT
NAME_LOGS=[]
PREFIX_IM="im"
PREFIX_HIST="hist"


#CLOUD THR VAL

cloudThresh = 0.2
dilationPixels = 3
erodePixels = 1.5
ndviThresh = -0.1
irSumThresh = 0.3
roi_ccp_max=0.1

## CONSTANTS FOR THE EVI COMPUTATION

DICT_EVI_PARAM={"L":1, "G":2.5, "C1":6, "C2":7.5}

#CONSTANT ABOUT MINMAX EXTRACTION OF THE S2 AND NDVI DATA
GEE_DRIVE_FOLDER="CSIRO/gee_data/"
DICT_TRANSLATE_BAND={"B2": "B", "B4": "R","B8": "NIR","B3":"G"}
NDVI_BAND=["B4","B8"]
EVI_BAND=["B4","B8","B2"]
NB_VI_CSV=2


#BURN SEVERITY dNDVI constant
