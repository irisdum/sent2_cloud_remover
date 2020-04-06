#!/bin/bash
# enable next line for debugging purpose
# set -x 

############################################
# User Configuration
############################################

# adapt this path to your needs
#export PATH=~/progs/snap/bin:$PATH
export PATH=/home/dum031/snap/bin:$PATH
gptPath="gpt -e"
export JAVA_OPTS="-Xmx8192m -XX:CompressedClassSpaceSize=256m"
export _JAVA_OPTIONS="-Xmx8192m -XX:CompressedClassSpaceSize=256m"
############################################
# Command line handling
############################################

# first parameter is a path to the graph xml
graphXmlPath="$1"

# second parameter is a path to a parameter file
parameterFilePath="$2"

# use third parameter for path to source products
sourceDirectory="$3"

# use fourth parameter for path to target products
targetDirectory="$4"

# the fifth parameter is a file prefix for the target product name, typically indicating the type of processing
targetFilePrefix="$5"

#the sixth parameter is the path to the WKT subset .txt file
wktFile="$6"
   
############################################
# Helper functions
############################################
removeExtension() {
    file="$1"
    echo "$(echo "$file" | sed -r 's/\.[^\.]*$//')"
}


############################################
# Main processing
############################################

# Create the target directory
mkdir -p "${targetDirectory}"

# the d option limits the elemeents to loop over to directories. Remove it, if you want to use files.
for F in $(ls -1d "${sourceDirectory}"/S1*.SAFE); do
  echo "$F"
  sourceFile="$(realpath "$F")"
  # During the preprocess we split the images on smaller tiles
  # shellcheck disable=SC1035
  i=1
  while IFS =read -r poly; do
    targetFilePrefix="${targetFilePrefix}"
    targetFile="${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${F})").dim"
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/vv_$(removeExtension "$(basename ${F})")_prepro}_$i" -PsourceBand=Amplitude_VV -Pgeometry="${poly}" -t  ${targetFile} ${sourceFile}
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/vh_$(removeExtension "$(basename ${F})")_prepro}_$i" -PsourceBand=Amplitude_VH -Pgeometry="${poly}"  -t  ${targetFile} ${sourceFile}
    gdalinfo "${targetDirectory}/vv_$(removeExtension "$(basename ${F})")_prepro.tif_$i"
    gdalinfo "${targetDirectory}/vv_$(removeExtension "$(basename ${F})")_prepro.tif_$i"
    i=$((i+1))
  done<"${wktFile}"

done

####test if the image works
#gdalinfo output4.tif
