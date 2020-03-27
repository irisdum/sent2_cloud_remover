#Constant to filter the results
DICT_COLLECTION={1: "COPERNICUS/S1_GRD", 2: "COPERNICUS/S2"}
S2_OPTPARAM={"instrument": "MSI", "processingLevel": "L1C"}
S1_OPTPARAM={"productType": "GRD", "sensorMode": "IW", "instrument": "C-SAR"} # "polarisation": "VH"
ORBIT_ID={1: "orbitNumber_start", 2: "MGRS_TILE"}

# Constant for the storing code
DIR_T=["date1/", "date2/"]
DIR_SENT=["sentinel1/", "sentinel2/"]
TEMPORARY_DIR= "TEMPORARY_DIR/"
DOWNLOAD_PATH= "./Downloads/" #path where the image are download from sara
SENT_FORMAT=[".tiff", "jp2"]
LISTE_BANDE_S2=["B04", "B03", "B02","B08"]
