pip install gdown
apt-get update
apt-get upgrade -y
apt-get install unzip

gdown --fuzzy https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view
gdown --fuzzy https://drive.google.com/file/d/1cJyPQzVOzsCZQctNBuHCqxHnOY7v7UiA/view

unzip -q dataset.zip
unzip -q Samples.zip

mkdir Dataset
mkdir Dataset/superresolution

mv dataset Dataset
mv Samples Dataset/superresolution

rm dataset.zip Samples.zip