pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view
gdown --fuzzy https://drive.google.com/file/d/1cJyPQzVOzsCZQctNBuHCqxHnOY7v7UiA/view

mkdir Dataset
mkdir Dataset/superresolution
mv dataset.zip Dataset
unzip Dataset/dataset.zip
mv Samples.zip Dataset/superresolution
unzip Dataset/superresolution/Samples.zip
mv Samples Dataset
mv dataset Dataset
