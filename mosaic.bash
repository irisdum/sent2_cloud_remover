#!/bin/bash

export PATH=/home/dum031/snap/bin:$PATH
gptPath="gpt -e"
export JAVA_OPTS="-Xmx8192m -XX:CompressedClassSpaceSize=256m"
export _JAVA_OPTIONS="-Xmx8192m -XX:CompressedClassSpaceSize=256m"

inputdir="$1"
outputdir="$2"

