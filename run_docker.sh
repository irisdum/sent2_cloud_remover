docker run -it --rm  -p 5000:80 -v  ~/code:/root/code -v  ~/projects/data61/:/root/data --workdir /root  osgeo/gdal:ubuntu-small-latest

pip install -r requirement.txt
