#!/bin/bash
set -x
export JAVA_OPTS="-Xmx4096m -XX:CompressedClassSpaceSize=256m"
export _JAVA_OPTIONS="-Xmx4096m -XX:CompressedClassSpaceSize=256m"
############################################
# User Configuration
############################################

# adapt this path to your needs
#export PATH=~/progs/snap/bin:$PATH
export PATH=/home/dum031/snap/bin:$PATH
gptPath="gpt -e"

############################################
# Command line handling
############################################

# first parameter is a path to the graph xml
graphXmlPath="./snap-confs/orbite.xml"

# second parameter is a path to a parameter file
parameterFilePath="./snap-confs/orbite.properties"

# use third parameter for path to source products
sourceDirectory="$3"

# use fourth parameter for path to target products
targetDirectory="./orbite"

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
  ${gptPath} ${graphXmlPath} -e -p ${parameterFilePath} -Dsnap.gpf.useFileTileCache=false -t ${targetFile} ${sourceFile}
done

