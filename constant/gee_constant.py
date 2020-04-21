#CONSTANT TO FILTER RESULTS TO FIND AND DOWNLOAD IMAGES
DICT_COLLECTION={1: "COPERNICUS/S1_GRD", 2: "COPERNICUS/S2"}
S2_OPTPARAM={"instrument": "MSI", "processingLevel": "L1C"}
S1_OPTPARAM={"productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR"} # "polarisation": "VH" "orbitDirection":"descending"
ORBIT_ID={1: "orbitNumber_start", 2: "MGRS_TILE"}

# STORING IMAGES CONSTANT
DIR_T=["date1/", "date2/"]
DIR_SENT=["sentinel1/", "sentinel2/"]
DOWNLOAD_PATH= "./Downloads/" # /datastore/dum031/data/dataset1/ #path where the image are download from sara
TEMPORARY_DIR= "temporary_dir/"
TILING_DIR="tiling_dir/"
XDIR="dataX/"
LABEL_DIR="label/"
SENT_FORMAT=["tiff", "jp2"]
LISTE_BANDE=[["vh","vv"],["B04", "B03", "B02","B08","cm"]]



### DOWNLOADING CRITERIA
FACTEUR_AREA=0.9 # correspond of the minimum area of the zone the sentinel 1 should cover

VAR_NAME="_prepro"

## TILING OPTION
OVERLAP=10  #overlap of the tiling

##PROJECTION SENTINEL 2
EPSG="EPSG:32756"


## DISPALY CONSTANT
BOUND_X=[100,1000]
BOUND_Y=[100,1000]

## CONVERT Uint16 2 Float 32
CONVERTOR=8000
SCALE_S1=1

## SCAN TILES THRESHOLD OF NUMBER OF CLOUD PIXEL ALLOWED TO KEEP THE TILE
CLOUD_THR=20