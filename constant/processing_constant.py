
# There should be some coherence with the way all the constant are defined :
#The elem of the list S1_BANDS and S2_BANDS should be the keys of DICT_GROUP_BAND_LABEL,DICT_GROUP_BAND_X,
# DICT_RESCALE_TYPE and DICT_SCALER
S1_BANDS="VV,VH".split(",")
S2_BANDS=["R,G,B,NIR"]
DICT_GROUP_BAND_LABEL={"R,G,B,NIR":[0,1,2,3]}
DICT_GROUP_BAND_X={"VV":[0,2],"VH":[1,3],"R,G,B,NIR":[4,5,6,7]}
DICT_RESCALE={"R":"","G":"center_norm11","B":"center_norm11","NIR":"center_norm11","VV":"centering",
              "VH":"centering"}
DICT_RESCALE_TYPE={"VV": "StandardScaler", "VH":"StandardScaler","R,G,B,NIR":"StandardScaler"}
DICT_SCALER={"VV": None, "VH":None,"R,G,B,NIR":None}