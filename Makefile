geojson_file=confs/train_kangaroo.geojson
wkt_file=confs/train_kangaroo_wkt.txt
grah_xml_sent1=snap-confs/calibrate_sent1_v2.xml
source_directory=/datastore/data/dataset2/
target_directory={source_directory}/prepro2/

geojson_2_wkt: ${wkt_file}
		@echo "We convert ${geojson_file} into ${wkt_file}"
		@python convert_geojson.py --input ${geojson_file} --output ${wkt_file}

convert_sent1:
	./processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date1/ ${target_directory}date1/ ${wkt_file}
	./processDatasetSent1.bash ${graph_xml_sent1} snap-confs/orbite.properties ${source_directory}date2/ ${target_directory}date2/ ${wkt_file}