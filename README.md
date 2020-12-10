# sent2-cloud-remover
IN PROGRESS, this code is not finished and change regularly !!!!!
This project aim at first using the method of conditional Generative adversial Network on Sentinel 1 and Sentinel 2 
in order to be able to recreate Sentinel 2 images from previous Sentinel2 and 1 and current Sentinel 1. Then assessing the quality of the
image produced studying the burned area maps. 
This project is thus divided in different steps : 
- Creation of the dataset (done, some clean of the code required)
- Training of the conditional Generative Adversial Network (done, some clean of the code required)
- Assessing  the results of the simulated image (in progress)  

Many of the different computation steps are easily available from the Makefile. 

## Installation requirements : 


 - Esa SNAP (command gpt will be used)
 The conda env yaml  is in env/clean_env.yml. It enables to use google earth engine api, gdal (2.4), buzzard (0.6), tensorfow2, 
 jupyter notebook, Fmask.
 Tensorboard does not seems to work within this environmnent.
 `conda-env create -f env/clean_env.yml -name proj_env`
 `conda activate proj_env`
 To run jupyter notebook within this conda env : 
 `python -m ipykernel install --user --name=myEnv`
 Then run `jupyter notebook` and select myEnv into the kernels available



## Downloading the data
The aim is to donwload the data which are going to be used for the training of the Generative Adversarial Network.
For each Sentinel 2 image downloaded we want to have the corresponding Sentinel 1 images (the closest in time).
## LOOKING AT THE AREA OF INTEREST
 TODO : include the Google earth engine scripts



## Downloading the data
The download of the image uses the library auscophub found https://github.com/CopernicusAustralasia/auscophub/releases`. 
It enables to send request to the GeoscienceAustralia/NCI Copernicus Hub server and download the images. 
The file `utils/download_image.py` gathers the functions useful to download the image. 

### The automatic version
(not recommended, not optimized and not truly recommended)
Input : 
- Name of the images 
- date range
- optional parameter filters
#### Finding cloud free images


Input : 
- an area (geojson you can use QGIS to create it)
- an date range
- A maximum ccp cloudy coverage percentage
- optional parameter filters
The script will connect to Google earth engine in order to find the Sentinel 2 images within the date range that have a lower
ccp thant max_ccp and cover the area. Given the date of acquisition of the Sentinel 2, the closest Sentinel 1 images (in time).
The python script used `find_images.py`, an exemple of the use is shown on `bash_scripts/search_image.sh`.
The cloud filter is python script adapted from  http://bit.ly/Sen2CloudFree
#### Run the download
Modify the variable used in the command `download_image` in the Makefile
Then run `make download_image
### The manual version for selecting Sentinel 2 data
First open google earth engine (GEE) and look at the intersted Sentinel 2 image. Select the area of interest as the 
intersection of the Sentinel 2 image and the Sentinel 1 footprint. Export it to your Google Drive (GEOJSON format)
 and visualize it with QGIS. Maybe modify the Bbox which you are going to use, then export it in two format : 
 - The UTM format used for your Sentinel 2 data. Currently, we work with UTM55S (East Australia)
 - The EPSG 4326 WSG 84 format
Place the exported geojson files into the confs directory, and modify in the Makefile the value of `geojson_file`,
 `geojson_utm_file`
Then from GEE, extract the id of the Sentinel 2 images you want to work with and replace in the Makefile the 
value of s2_im_t1, s2_im_t0
Then run : `download_images_from_s2name`

## Preprocessing using SNAP gpt. 

### Requirements
- The area geojson `confs/dataset2/dataset2_bbox_utm55s.geojson`
- The directory with Sentinel 1 and Sentinel 2 images.
- - For Sentinel 1 : `new_processDatasetSent1.bash`, `snap-confs/calibrate_sent1_zs_utm55s.xml` or `snap-confs/calibrate_sent1_zs.xml`   
   - For Sentinel 2 : `new_processDatasetSent1.bash`, `snap-confs/calibrate_sent2_zs.xml`

You may need to adapt the variable definition in the bash scripts, depending on how you installed SNAP. Pay attention to 
PATH and gptPath variable. gptPath corresponds to the path where gpt executable is located. PATH needs to be updated with
the location of snap/bin

### Preprocessing

The pipeline of the preprocessing is in the .xml files. 


### bash script

If there are numerous geometry in the .txt file For each geometry, the preprocessing of is done for each bands 
(B2,B3,B4,B8) of Sentinel 2 and on both bands of Sentinel 1 VV,VH. 
Run `make convert_sent1` to preprocess Sentinel 1 images, Run `make convert_sent2`to preprocess the Sentinel 2 data.
The code have been ran on HPC, however it might be interesting to modify some parameters 
`gpt -c CACHE_NUMBER -q PARALLELISM` of gpt in the bash scripts : new_processDatasetSent1.bash an new_processDatasetSent2.bash
The combination found might not be the best one, the code is still really slow. 

## Tiling
`make test_tiling`

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

## Training a Model
If you are only interested into the training 

First you can use the specific training condo environment : env/training_env.yaml : 

```bash
conda env create -f env/training_env.yaml
conda activate training_env
python -m ipykernel install --user --name=training_env
```

### Start a training

#### With Jupyter

Start jupyter notebook. If in a remote machine : add  `--ip=0.0.0.0 --no-browser`

Now you can open the jupyter Notebook : notebooks/Trainings.ipynb

Modify the constant, defined at the beginning at the notebook and run the training. 

#### As a batch job

The cycle GAN model used is defined in models/clean_gan.py
In order to train a model two yaml should be modified, examples available in GAN_confs :  

- model.yaml
- train.yaml
  Then running gan_train.sh path_to_model_yaml path_to_train_yaml will start the training job

### Supervise the training

#### Tensorboard : metric supervision

The training is also configured to be supervised using Tensorboard. 

Open a new terminal window within training_env and run tensorboard.

Local command `tensorboard --logdir <path>`

If in remote machine : `tensorboard --logdir <path> --host 0.0.0.0`

#### Validation image visualization : notebook

Eventually you can also look at how the images look like during the training. 

Open the notebook *SPECIFY NOTEBOOK*

Notebooks will soon be released showing examples of the results

### Normalization of the data

Before training the model, it is recommended to rescale your data between -1 and 1.

Currently we have been downloading Sentinel 2 Level 2A products, and Sentinel 1 GRD product coming from *GeoscienceAustralia/NCI Copernicus Hub server*. Moreover Sentinel 1 images have been preprocessed using snap gpt tool (check the *snap-confs/calibrate_sent1_zs_utm55s.xml* file) and the preprocessing section for more details. 

However after the preprocessing some values of Sentinel 1 can be negative, and as we want to work with db intensities, we fix this issue using a knn algorithm. All the pixels having negative values of the SAR images are replaced using their neighbour intensities. Then we apply 10*log10 transformation before applying StandardScaler (scikit-learn method).

## Vegetation index 
In order to assess the burned area different vegetation index have been implemented.

## Land classification

# Analyszing the GAN output

1) Run a prediction on a directory use gan_predict.sh 
it will create a directory with the predicted image saved. This output will depend of the model trained and the iteration the model has been stopped

2) Recreate the original image : using mosaic_pred.py : it will for each tile find its fp and recreate the global image 









 



 
