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
- an area (geojson you can use QGIS to create it)
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
used to download the images. The script that gathers both action is `run_download_images.py`.  
An exemple of the use of this script is shown in `download_images.sh`

## Storing the data

In order to build or dataset, the images download needs to be stored in different files depending on their download date.
The functions used are in `store_data.py`

## Preprocessing using SNAP gpt. 

### Requirements
- The area geojson `confs/train_kangaroo.geojson`
- The directory with Sentinel 1 and Sentinel 2 images.  
- - For Sentinel 1 : `processDatasetSent1.bash`, `snap-confs/calibrate_sent1.xml` or `snap-confs/calibrate_sent1_v2.xml`   
   - For Sentinel 2 : `processDatasetSent1.bash`, `snap-confs/calibrate_sent2.xml`

First the geojson needs to be converted in .txt file, where the area is written as WKT format. If there are many polygons 
in the geojson, in the .txt file one line = one polygon in WKT format.  
Thus RUN `python convert_geojson.py --input input_zone.geojson --output output_zone_wkt.txt`

You may need to adapt the variable definition in the bash scripts, depending on how you installed SNAP. Pay attention to 
PATH and gptPath variable. gptPath corresponds to the path where gpt executable is located. PATH needs to be updated with
the location of snap/bin

### Preprocessing

The pipeline of the preprocessing is in the .xml files. 

For Sentinel 1 : 

- Apply Orbit File
- Subset 
- Calibration
- Speckle (Refined Lee)
- Terrain Correction
- Write as TIFF

xml v2 : 
- Apply Orbit File
- Subset 
- Thermal Noise
- Remove GRD Noise
- Calibration
- Speckle (Refined Lee)
- Terrain Correction
-Linear from db
- Write as TIFF
For Sentinel 2 : 
- Resample at 10 meters
- Subset 
- Write TIFF

### bash script

If there are numerous geometry in the .txt file For each geometry, the preprocessing of is done for each bands 
(B2,B3,B4,B8) and (VV,VH).

## Build dataset 
### Requirements
The preprocessed image are stored in two different directories depending on their date range. 

Once the images are preprocessed, it is now time to sort the data in order to split the training data and the label in two
different directories : 
- dataX: contains Sentinel 1 and 2 at the first date range and Sentinel 1 at the second date range  
    - Sentinel1_t0 : directory with Sentinel 1 images 
    - Sentinel1_t1
    - Sentinel2_t0 
- label: contains Sentinel 2 at the second date range

For dataX : 
Using gdalbuildvrt a VRT file is created creating first a mosaic for each image bands. Then another VRT file is made merging
the required bands together : 





 



 
