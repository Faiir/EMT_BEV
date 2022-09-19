#! /bin/bash 

echo "Starting Installation"


echo "Colab? y/n:"  
read colab

# Linux
wget=/usr/bin/wget
WORK_DIR  = ./

map_expansion = "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/nuScenes-map-expansion-v1.3.zip?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=sVge3r41w66xRxJlbyNaMR7hUXE%3D&Expires=1664004309"
mini_nuscenes = "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-mini.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=4NjqwjxAAS8rNloDRjXfB%2Fg3fbs%3D&Expires=1664004394"

echo "Downloading map-expansion"
wget -O "nuScenes-map-expansion-v1.3.zip" map_expansion
echo "Downloading mini-dataset"
wget -O "v1.0-mini.tgz" mini_nuscenes

echo "unziping mini-dataset"
unzip -d ./data/ v1.0-mini.tgz
echo "unziping map-expansion"
unzip -d ./data/ nuScenes-map-expansion-v1.3.zip

if colab == "y";
then
python -m venv ./.venv
source ./venv/bin/activate
fi
echo "starting env installation"
pip uninstall torch torchvision torchaudio
pip install wheels
pip install requirements_linux.txt


python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini

python setup.py develop