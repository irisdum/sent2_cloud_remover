

S1_BANDS="VV,VH".split(",")
S2_BANDS="R,G,B,NIR".split(",")
DICT_GROUP_BAND_LABEL={"R,G,B,NIR":[0,1,2,3]}
DICT_GROUP_BAND_X={"VV":[0,2],"VH":[1,3],"R,G,B,NIR":[4,5,6,7]}
DICT_RESCALE={"R":"center_norm11","G":"center_norm11","B":"center_norm11","NIR":"center_norm11","VV":"centering",
              "VH":"centering"}