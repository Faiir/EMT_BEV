#! /bin/bash 

echo "Starting Installation"


# echo "Colab? y/n:"  
# read colab

# Linux
#wget=/usr/bin/wget
WORK_DIR  = ./

map_expansion = "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/nuScenes-map-expansion-v1.3.zip?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=sVge3r41w66xRxJlbyNaMR7hUXE%3D&Expires=1664004309"
mini_nuscenes = "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-mini.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=4NjqwjxAAS8rNloDRjXfB%2Fg3fbs%3D&Expires=1664004394"

echo "Downloading map-expansion"
wget -O "nuScenes-map-expansion-v1.3.zip" map_expansion
echo "Downloading mini-dataset"
wget -O "v1.0-mini.tgz" https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-mini.tgz?AWSAccessKeyId=ASIA6RIK4RRMHYFNG7DA&Signature=zWmpOLOblYR4Le%2BN0dK%2BimpFse8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEKH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIAdUyrq1qFGYwe7irRgEiOI9UbvQs6oGNdHc4iEoWbTrAiEAkaal9wfF%2F2CFWQ9FiutRN0Tb%2BmSo4VEn6cZZWU3KXj8q9AIIeRADGgw5OTkxMzk2MDk2ODgiDBzKYUKT9cD%2BRP9eWCrRAlNztQG3wvArFRLrrAoMvw5Dc%2F2os7grOQ6HNEoogo%2BPZhA6Iu%2BQXubesYM3S1WzAx0%2BILXkLZskcyG%2FZ5%2BSz4C1Fpx61GlzaES2c4a%2BrDaj12aOAB4KHPSyo0269enBL2cnXHp0zGTaUaq1T3pOkFf6IlFhs7saxGRinv0K2m8JoNnKjSd%2FUzRWKYlJ%2BkxJbp4udM1BSe6vm%2FwjmTcJXbc9SkAisTkNA2Fz0aI1RwozxC6wHmlDJNKnJH74%2FeKDQoAHA1Ac6nQl8qhJ3cOF5oejeJJKTV60xiY0Czn5zGtE088xyDFm2GwV14aVpDoY%2F5xwOI2x3ZqyDx4IB%2FavbqD903mizINMEsddnEGl1ZL0AW4CXVEgoIAz9k0bPUcs%2B5Un72oyMUW4AVOzeQ01adNUOwzItPyKoIk8LGr7iMebeE8DT6TvsH2Qm9O3y8B9j5owtOqWoQY6ngE156bcYfEGW%2BzPK9VJTLsIWRY8oE6A%2BY8shcM0eZdcAqSNF11O6PN%2Be6mUi3P2cYlmCSfPjXviKqk1Ab964EEMHjlBDtm%2Fb%2BDO2IhjKvm1oUe7FYwA65yrNm632uVRV2kEH0OSYKZAZa3gfHLuNufnICsPw1JEOHowmz%2B%2Bhod9J7QxYqS1LDSbmERuyKDNm9yACDW4uvYvxo1iJW%2FAJw%3D%3D&Expires=1680624822

echo "unziping mini-dataset"
tar zxvf v1.0-mini.tgz -C ./data/nuscenes 
echo "unziping map-expansion"
unzip -d ./data/nuscenes/maps/ nuScenes-map-expansion-v1.3.zip

if colab == "y";
then
python -m venv ./.venv
source ./venv/bin/activate
fi
echo "starting env installation"
pip uninstall -y torch torchvision torchaudio torchtext resampy tensorflow jax
pip install wheels
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

python setup.py develop

python /content/EMT_BEV/tools/create_data.py nuscenes --root-path /content/EMT_BEV/data/nuscenes --out-dir /content/EMT_BEV/data/nuscenes --extra-tag nuscenes --version v1.0-mini
python /home/niklas/future_instance_prediction_bev/EMT_BEV/tools/create_data.py nuscenes --root-path /home/niklas/future_instance_prediction_bev/EMT_BEV/data/nuscenes --out-dir /home/niklas/future_instance_prediction_bev/EMT_BEV/data/nuscenes --extra-tag nuscenes --version v1.0-test