import zipfile

import os.path

from kaggle.api.kaggle_api_extended import KaggleApi

# Write supporting functions here



def download_files(api, files_to_download, storage_location):
    
    #Loop over the files
    for file in files_to_download:

        #Send download request to kaggle api for that file from the amex
        #competition
        #This method automatically checks if you already have downloaded
        #the most recent version of the data
        api.competition_download_file(
            competition='amex-default-prediction',
            file_name=file,
            path=storage_location
        )

        #Check if we already have extracted the csv from the zip
        if not os.path.isfile(storage_location + '/' + file): 

            path_to_zip_file = storage_location + '/' + file + '.zip'

            #Open the zip file in read mode
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(storage_location)
    

def run():
    print("Acquire: downloading raw data files...")
    
    #Create api instance object
    api = KaggleApi()
    #Run authentication method so that the instance is authenticated
    api.authenticate()
    
    storage_location = 'data/raw'
    
    files_to_download = ['train_labels.csv', 'train_data.csv']
    
    download_files(api, files_to_download, storage_location)
    
    print("Acquire: Completed!")

    
def import_train():
    
    base_url = '../../data/prepared/'
    
    train = pd.read_csv(base_url + 'train_data.csv')
    
    train_labels = pd.read_csv(base_url + 'train_labels.csv')
    
    return train, train_labels


def import_validate():
    
    base_url = '../../data/prepared/'
    
    valid = pd.read_csv(base_url + 'val_data.csv')
    
    valid_labels = pd.read_csv(base_url + 'val_labels.csv')
    
    return valid, valid_labels