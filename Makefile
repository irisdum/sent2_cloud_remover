geojson_file=confs/train_kangaroo.geojson
geojson_utm_file=confs/train_kangaroo_utm2.geojson
wkt_file=confs/train_kangaroo_wkt2.txt
graph_xml_sent1=snap-confs/calibrate_sent1_v2.xml
source_directory=/datastore/dum031/data/dataset2/
target_directory=${source_directory}prepro2/
build_dataset_dir=${target_directory}build_dataset/
graph_xml_sent2=snap-confs/calibrate_sent2.xml
geojson_2_wkt:
		@echo "We convert ${geojson_file} into ${wkt_file}"
		@python convert_geojson.py --input ${geojson_file} --output ${wkt_file}

convert_sent1:
		@echo "Starting preprocessing Sentinel 1"
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
		./new_processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}
convert_sent2:
	@echo "Starting preprocessing Sentinel 2"
	.new_processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date1 ${target_directory}date1 ${wkt_file}
	./new_processDatasetSent2.bash ${graph_xml_sent2} snap-confs/orbite.properties ${source_directory}date2 ${target_directory}date2 ${wkt_file}

start_env_processing:
	@conda activate myenv

quit_env_processing:
	@conda deactivate

tiling:
	@python processing.py --input_dir ${target_directory} --output_dir ${build_dataset_dir} --geojson ${geojson_utm_file}

