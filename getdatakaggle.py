#import os
#os.environ['KAGGLE_USERNAME'] = 'sebastian180897'
#os.environ['KAGGLE_KEY'] = 'ec2cf92048706a5225294bac6aa2a93c'
#from kaggle.api.kaggle_api_extended import KaggleApi
#import zipfile
#api = KaggleApi()
#api.authenticate()
# Download the competition files
#competition_name = 'playground-series-s4e6'
#download_path = 'data/'
#api.competition_download_files(competition_name, path=download_path)
# Unzip the downloaded files
#for item in os.listdir(download_path):
  #if item.endswith('.zip'):
   #zip_ref = zipfile.ZipFile(os.path.join(download_path, item), 'r')
   #zip_ref.extractall(download_path)
   #zip_ref.close()
   #print(f"Unzipped {item}")