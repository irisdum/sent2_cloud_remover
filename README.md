# sent2-cloud-remover
IN PROGRESS, this code is not finished  

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

Input : 
- an area (geojson)
- an date range
- A maximum ccp cloudy coverage percentage
- optional parameter filters
The script will connect to Google earth engine in order to find the Sentinel 2 images within the date range that have a lower
ccp thant max_ccp and cover the area. Given the date of acquisition of the Sentinel 2, the closest Sentinel 1 images (in time).
The python script used `find_images.py`, an exemple of the use is shown on `bash_scripts/search_image.sh`.


## Downloading the data

Input : 
- Name of the images 
- date range
- optional parameter filters

The download of the image uses the library auscophub found https://github.com/CopernicusAustralasia/auscophub/releases`. 
It enables to send request to the GeoscienceAustralia/NCI Copernicus Hub server and download the images. 
The file `utils/download_image.py` gathers the functions useful to download the image. 

Eventually to download the image, first their name is found using the find_image.py script. The output of this script is then 
used to download the images. 




 
