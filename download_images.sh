python3 run_download_images.py --bd1 "2019-10-15" --ed1 "2019-12-30" --bd2 "2020-01-10" --ed2 "2020-03-25" --sent2criteria "lessclouds" --zone "./confs/train_kangaroo.geojson" --ccp 5 --save false --output_path "./test_image/"  --shp  "./confs/fp_kangaroo.shp"

### COMMAND LINE TO STORE WELL THE DOWNLOAD IMAGES


cp ./download_image.sh  /home/dum031/data/dataset1/download_image.sh
cp ./gee_constant.py /home/dum031/data/dataset1/gee_constant.py