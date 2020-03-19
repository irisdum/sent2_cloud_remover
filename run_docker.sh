 docker run -it --rm --mount "type=bind,src=$(pwd),dst=/root/code/sent2-cloud-remover"  --workdir /root/code/sent2-cloud-remover  osgeo/gdal:ubuntu-small-latest
 pip install -r requirement.txt