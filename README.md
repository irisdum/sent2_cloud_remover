# sent2-cloud-remover
Use of multi-temporal SAR data and optical image with the use of cGAN to create synthetise cloud free optical image
This repesitory describe all the scripts used to find the cloud free sentinel 2 images and their corresponding sentinel 1,
download the data, preprocess them using SNAP, creating the tiles (256*256) for the data and for the label.


## Installation requirements : 
 - Google earth engine
 - Esa SNAP (command gpt will be used)
 - GDAL 
 Exemple of a docker image with GDAL : 
 To build the docker image
`docker build -t thinkwhere/gdal-python:3.7-ubuntu . `  
To run the docker image
` docker run -p 8888:8888  `

## Downloading the data

### Finding cloud free images

The aim of the script is given a criteria ccp (ccp cloudy coverage percentage), and a date range finding the image Sentinel 2
images with a a ccp lower than the constant

 
