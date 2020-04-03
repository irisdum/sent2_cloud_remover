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
  sourceFile="$(realpath "$F")"
  targetFile="${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${F})").dim"
  file_name_vv="vv_${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${F})")_prepro"
  file_name_vh="vh_${targetDirectory}/${targetFilePrefix}_$(removeExtension "$(basename ${F})")_prepro"
  ${gptPath} ${graphXmlPath} -e -p "vv_${parameterFilePath}"  -Pfile=file_name_vv  -t ${targetFile} ${sourceFile}
  ${gptPath} ${graphXmlPath} -e -p "vh_${parameterFilePath}"  -Pfile=file_name_vh  -t ${targetFile} ${sourceFile}
done

####test if the image works
gdalinfo output4.tif
