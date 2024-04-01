pip install gdown
apt-get update
apt-get upgrade -y
apt-get install unzip
gdown --fuzzy https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view
gdown --fuzzy https://drive.google.com/file/d/1cJyPQzVOzsCZQctNBuHCqxHnOY7v7UiA/view
gdown --fuzzy https://drive.google.com/file/d/1uJmDZw649XS-r-dYs9WD-OPwF_TIroVw/view
unzip -q dataset.zip
unzip -q Samples.zip
unzip -q Dataset.zip
mkdir Data
mkdir Data/Classification
mkdir Data/Diffusion
mkdir Data/Superresolution
mv dataset Data/Classification
mv Samples Data/Diffusion
mv Dataset Data/Superresolution
rm dataset.zip Samples.zip Dataset.zip
gdown --fuzzy https://drive.google.com/file/d/1plYfM-jFJT7TbTMVssuCCFvLzGdxMQ4h/view
unzip "Dataset 3B.zip"
mv Dataset Data/Task2B
rm "Dataset 3B.zip"