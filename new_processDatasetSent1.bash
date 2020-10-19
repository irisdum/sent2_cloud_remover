#!/bin/bash
################################
# adapt this path to your needs

export PATH=$PATH:/srv/osirim/idumeur/snap/bin

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


#the sixth parameter is the path to the WKT subset .txt file
wktFile="$5"

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
for F in $(ls -1d "${sourceDirectory}"/S1*.zip); do
  # echo "$F"
  sourceFile="$(realpath "$F")"
  # During the preprocess we split the images on smaller tiles
  # shellcheck disable=SC1035
  echo "${sourceFile}"
  echo "${wktFile}"
  i=1
  while IFS= read -r poly; do
    targetFilePrefix="process1_${i}"
    targetFile="${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${F})").dim"
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/$(removeExtension "$(basename ${F})")_prepro_$i" -Pgeometry="${poly}" -t  ${targetFile} -PinputFile=${sourceFile}
  done<"${wktFile}"

done

####test if the image works
#gdalinfo output4.tif
