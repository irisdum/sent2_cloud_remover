#Constant to filter the results
DICT_COLLECTION={1: "COPERNICUS/S1_GRD", 2: "COPERNICUS/S2"}
S2_OPTPARAM={"instrument": "MSI", "processingLevel": "L1C"}
S1_OPTPARAM={"productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR"} # "polarisation": "VH" "orbitDirection":"descending"
ORBIT_ID={1: "orbitNumber_start", 2: "MGRS_TILE"}

# Constant for the storing code
DIR_T=["date1/", "date2/"]
DIR_SENT=["sentinel1/", "sentinel2/"]
DOWNLOAD_PATH= "./Downloads/" # /datastore/dum031/data/dataset1/ #path where the image are download from sara
TEMPORARY_DIR= "temporary_dir/"
SENT_FORMAT=["tiff", "jp2"]
LISTE_BANDE=[["vh","vv"],["B04", "B03", "B02","B08"]]

FACTEUR_AREA=0.9
