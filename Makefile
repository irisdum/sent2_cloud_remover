begin_date1=2019-11-03
end_date1=2019-11-10
begin_date2=2020-02-01
end_date2=2020-02-06
begin_date3=2019-09-25
end_date3=2019-09-28
CCP=50
geojson_file=confs/dataset2/dataset2_bbox_wsg84.geojson
geojson_utm_file=confs/dataset2/d2_small_forest_utm55s.geojson
wkt_file=confs/train_kangaroo_wkt.txt
graph_xml_sent1=snap-confs/calibrate_sent1_zs_utm55s.xml
source_directory=/srv/osirim/idumeur/data/dataset6/
target_directory=${source_directory}prepro1/
build_dataset_dir=${target_directory}small_forest_build_dataset/
build_dataset_landclass=${source_directory}build_dataset_landclass/
graph_xml_sent2=snap-confs/calibrate_sent2_zs.xml
output_split_dir_name=input_forest_dataset
split_test=0.15
split_train=0.80
split_val=0.05
clouds=True
path_train_yaml=GAN_confs/train.yaml
path_model_yaml=GAN_confs/model.yaml
path_grid_geosjon=${build_dataset_dir}dataX/Sentinel1_t0/tiling_sent1_t0_fp.geojson
s2_im_t1=S2B_MSIL2A_20200204T000239_N0213_R030_T55HGB_20200204T015232
s2_im_t0=S2B_MSIL2A_20191106T000239_N0213_R030_T55HGB_20191106T014722
s2_im_t3=S2B_MSIL2A_20190927T000239_N0213_R030_T55HGB_20190927T015503
training_dir=/srv/osirim/idumeur/trainings/new_model_corr/
training_number=31
pred_dataset=${target_directory}${output_split_dir_name}/test/
weight=295
pref_pred_image=tr${training_number}_w_${weight}_test_d6
output_mosaic_dir=${target_directory}pred_tr${training_number}_i${weight}/
snap_property_file=/srv/osirim/idumeur/snap/etc/snap.properties

download_images_from_s2name_2d:
	@python run_download_images.py --bd ${begin_date1} ${begin_date2} --ed ${end_date1}  ${end_date2} --sent2criteria "lessclouds" --zone ${geojson_file} --ccp ${CCP} --save true --output_path ${source_directory}   --s2_names ${s2_im_t0} ${s2_im_t1}

download_images_from_s2name_3d:
		@python run_download_images.py --bd ${begin_date3} ${begin_date1} ${begin_date2} --ed ${end_date3} ${end_date1}  ${end_date2} --sent2criteria "lessclouds" --zone ${geojson_file} --ccp ${CCP} --save true --output_path ${source_directory}   --s2_names ${s2_im_t3} ${s2_im_t0} ${s2_im_t1}

conda_rasterio:
	conda config --add channels conda-forge
	conda install rasterio

find_image:
	@python3 find_image.py  --bd ${begin_date1} --ed ${end_date1} --zone ${geojson_file} --c 2

get_s2_minmax:
	python gee_ndvi_minmax.py --path_bdata ${build_dataset_dir} --path_input_data ${target_directory}${output_split_dir_name} --bd 2019-01-01 --ed 2019-12-31 --vi s2 --export GEE

get_ndvi_minmax:
	python gee_ndvi_minmax.py --path_bdata ${build_dataset_dir} --path_input_data ${target_directory}${output_split_dir_name} --bd 2019-01-01 --ed 2019-12-31 --vi ndvi --export GEE --path_csv ${source_directory}

get_evi_minmax:
	python gee_ndvi_minmax.py --path_bdata ${build_dataset_dir} --path_input_data ${target_directory}${output_split_dir_name} --bd 2019-01-01 --ed 2019-12-31 --vi evi --export GEE --path_csv ${source_directory}

download_image:
		@python run_download_images.py --bd1 ${begin_date1} --ed1 ${end_date1} --bd2 ${begin_date2} --ed2 ${end_date2} --sent2criteria "lessclouds" --zone ${geojson_file} --ccp ${CCP} --save false --output_path ${source_directory}

geojson_2_wkt:
		@echo "We convert ${geojson_file} into ${wkt_file}"
		@python convert_geojson.py --input ${geojson_file} --output ${wkt_file}

convert_sent1:
		@echo "Starting preprocessing Sentinel 1"
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}

convert_sent1_d3:
		@echo "Starting preprocessing Sentinel 1"
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date3 ${target_directory}date3 ${wkt_file}

convert_sent2:
	@echo "Starting preprocessing Sentinel 2"
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}

convert_sent2_d3:
	@echo "Starting preprocessing Sentinel 2"
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}
	./processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date3 ${target_directory}date3 ${wkt_file}

tiling:
	@python processing.py --input_dir ${target_directory} --output_dir ${build_dataset_dir} --geojson ${geojson_utm_file}

test_tiling:
	@python tiling.py --input_dir ${target_directory} --output_dir ${build_dataset_dir} --geojson ${geojson_utm_file}


tiling_overlap:
	@python processing.py --input_dir ${target_directory} --output_dir ${build_dataset_dir} --geojson ${geojson_utm_file} --overlap 50

split_dataset:
	@python split_dataset.py --input_dataset ${build_dataset_dir} --output_dir_name ${output_split_dir_name} --ptest ${split_test} --pval ${split_val} --ptrain ${split_train} --keep_clouds ${clouds}
	zip -r ${target_directory}output_dir_name ${output_split_dir_name}.zip ${target_directory}output_dir_name ${output_split_dir_name}
train_model:
	python train.py --model_path ${path_model_yaml} --train_path ${path_model_yaml}

train_on_cluster:
	oarsub -q production "nodes=1/gpu=2,walltime=25:00:00" -p "cluster='grimani'" ./gri5000_train.sh ${path_model_yaml} ${path_model_yaml} ${target_directory}${output_split_dir_name}

install_snap:
	@wget http://step.esa.int/downloads/7.0/installers/esa-snap_all_unix_7_0.sh
	chmod +x esa-snap_all_unix_7_0.sh
	./esa-snap_all_unix_7_0.sh
	@echo "Then run PATH=$PATH:path_to_snap_install/bin"

download_to_split:
	download_image
	geojson_2_wkt
	convert_sent1
	convert_sent2
	tiling
	split_dataset

download_aus18_tif:
	wget https://data.gov.au/dataset/6ebffd6f-a937-4fa6-843a-3fde2effbacd/resource/83f98691-de14-4c7d-a3f0-e445fba0b4c7/download/aus_for18.tiff
	mv aus_for18.tiff ${source_directory}
	gdalwarp -t_srs EPSG:32756 ${source_directory}aus_for18.tiff ${source_directory}aus_for18_reproj.tiff

predict:
	sbatch gan_predict.sh ${training_dir} ${training_number} ${weight} ${pred_dataset} ${pref_pred_image} ${source_directory}

predict_val:
	sbatch gan_predict_val.sh ${training_dir} ${training_number} ${weight}

mosaic_predictions:
	python mosaic_pred.py --bd_dir ${build_dataset_dir} --pred_dir ${training_dir}training_${training_number}/image_${pref_pred_image}_iter_${weight}/ --out_dir ${output_mosaic_dir} --im_pref tr${training_number}_i${weight}

mosaic_gt:
	python mosaic_pred.py --bd_dir ${build_dataset_dir}  --pred_dir ${pred_dataset}label/ --out_dir ${target_directory}mosaicgt/ --im_pref label_tr${training_number}_i${weight} --path_csv ${source_directory}

help:
	@echo "[HELP] "
	@echo "To Download the image run make download_image"
	@echo "To convert geojson to wkt_txt format make geojson_2_wkt"
	@echo "To apply SNAP pipeline defined on xml for Sentinel i run  convert_senti, replace i by 1 or 2"
	@echo "To apply the tiling process coordinates system conversion, resampling 10 res .. make tiling"
	@echo "To split the dataset between train, val and test folder make split_dataset"
	@echo "To train a model make train_model, please check the value of the train yaml and model yaml before to do so, it will run with sbatch !"
	@echo "To download land classfication tiff from Australian forest report 2018 make download_aus18_tif and reroj it to EPSG 32756"
	@echo "To get the min max value from sentinel 2 over a year make get_s2_minmax:"
