import zipfile

import os.path

from kaggle.api.kaggle_api_extended import KaggleApi

# Write supporting functions here

def run():
    print("Acquire: downloading raw data files...")
    #Create api instance object
    api = KaggleApi()
    #Run authentication method so that the instance is authenticated
    api.authenticate()
    
    storage_location = 'data/raw'
    
    files_to_download = ['train_labels.csv', 'train_data.csv']

    for file in files_to_download:

        api.competition_download_file(
            competition='amex-default-prediction',
            file_name=file,
            path=storage_location
        )

        if not os.path.isfile(storage_location + '/' + file): 

            path_to_zip_file = storage_location + '/' + file + '.zip'

            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(storage_location)
    print("Acquire: Completed!")
