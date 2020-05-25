begin_date1=2019-10-15
end_date1=2019-12-30
begin_date2=2020-02-01
end_date2=2020-04-01
CCP=2
geojson_file=confs/train_kangaroo.geojson
geojson_utm_file=confs/train_kangaroo_utm2.geojson
wkt_file=confs/train_kangaroo_wkt2.txt
graph_xml_sent1=snap-confs/calibrate_sent1_v2.xml
source_directory=/datastore/dum031/data/dataset2/
target_directory=${source_directory}prepro2/
build_dataset_dir=${target_directory}build_dataset/
graph_xml_sent2=snap-confs/calibrate_sent2.xml
output_split_dir_name=input_large_dataset
split_test=0.15
split_train=0.80
split_val=0.05
path_train_yaml=GAN_confs/train.yaml
path_model_yaml=GAN_confs/model.yaml

conda_rasterio:
	conda config --add channels conda-forge
	conda install rasterio

find_image:
	@python3 find_image.py  --bd ${begin_date1} --ed ${end_date1} --zone ${geojson_file} --c 2

get_s2_minmax:
	@python gee_ndvi_minmax.py --path_bdata ${build_dataset_dir} --path_input_data ${target_directory}${output_split_dir_name} --bd 2019-01-01 --ed 2019-12-31 --vi s2
get_ndvi_minmax:
	@python gee_ndvi_minmax.py --path_bdata ${build_dataset_dir} --path_input_data ${target_directory}${output_split_dir_name} --bd 2019-01-01 --ed 2019-12-31 --vi ndvi

download_image:
		@python run_download_images.py --bd1 ${begin_date1} --ed1 ${end_date1} --bd2 ${begin_date2} --ed2 ${end_date2} --sent2criteria "lessclouds" --zone ${geojson_file} --ccp ${CCP} --save false --output_path ${source_directory}  --shp  "../confs/fp_kangaroo.shp"

geojson_2_wkt:
		@echo "We convert ${geojson_file} into ${wkt_file}"
		@python convert_geojson.py --input ${geojson_file} --output ${wkt_file}

convert_sent1:
		@echo "Starting preprocessing Sentinel 1"
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}
convert_sent2:
	@echo "Starting preprocessing Sentinel 2"
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}

start_env_processing:
	@conda activate myenv
quit_env_processing:
	@conda deactivate

create_env_processing:
	@conda config --add channels conda-forge
	@conda config --set channel_priority strict
	@conda create -n myenv python-fmask

convert_sent_inconda:
	start_env_processing
	convert_sent2
	convert_sent1
	quit_env_processing

tiling:
	@python processing.py --input_dir ${target_directory} --output_dir ${build_dataset_dir} --geojson ${geojson_utm_file}

split_dataset:
	@python split_dataset.py --input_dataset ${build_dataset_dir} --output_dir_name ${output_split_dir_name} --ptest ${split_test} --pval ${split_val} --ptrain ${split_train}

train_model:
	@sbatch gan_train.sh ${path_model_yaml}  ${path_train_yaml}

install_snap:
	@echo "Not implemented yet"

download_to_split:
	download_image
	geojson_2_wkt
	convert_sent_inconda
	tiling
	split_dataset


help:
	@echo "[HELP] "
	@echo "To Download the image run make download_image"
	@echo "To convert geojson to wkt_txt format make geojson_2_wkt"
	@echo "To apply SNAP pipeline defined on xml for Sentinel i run  convert_senti, replace i by 1 or 2"
	@echo "To apply SNAP pipeline within the conda env run make convert_sent_inconda"
	@echo "To apply the tiling process coordinates system conversion, resampling 10 res .. make tiling"
	@echo "To split the dataset between train, val and test folder make split_dataset"
	@echo "To train a model make train_model, please check the value of the train yaml and model yaml before to do so, it will run with sbatch !"
