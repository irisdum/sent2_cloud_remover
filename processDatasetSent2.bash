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


#the sixth parameter is the path to the WKT subset .txt file
wktFile="$5"


############################################
# Helper functions
############################################
removeExtension() {
    file="$1"
    echo "$(echo "$file" | sed -r 's/\.[^\.]*$//')"
}


# Create the target directory
mkdir -p "${targetDirectory}"

# the d option limits the elemeents to loop over to directories. Remove it, if you want to use files.
for F in $(ls -1d "${sourceDirectory}"/S2*.SAFE); do
  # echo "$F"
  sourceFile="${F}"
  # During the preprocess we split the images on smaller tiles
  # shellcheck disable=SC1035
  echo "${sourceFile}"
  echo "${wktFile}"
  i=1
  while IFS= read -r poly; do
    targetFilePrefix="process1_${i}"
    targetFile="${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${F})").dim"
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/b2_$(removeExtension "$(basename ${F})")_prepro}" -PBands="B2" -Pgeometry="${poly}" -t  ${targetFile} ${sourceFile}
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/b3_$(removeExtension "$(basename ${F})")_prepro}" -PBands="B3" -Pgeometry="${poly}" -t  ${targetFile} ${sourceFile}
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/b4_$(removeExtension "$(basename ${F})")_prepro}" -PBands="B4" -Pgeometry="${poly}" -t  ${targetFile} ${sourceFile}
    ${gptPath} ${graphXmlPath} -e -p "${parameterFilePath}"  -Pfile="${targetDirectory}/b8_$(removeExtension "$(basename ${F})")_prepro}" -PBands="B8" -Pgeometry="${poly}" -t  ${targetFile} ${sourceFile}
    i=$((i+1))
  done<"${wktFile}"

done