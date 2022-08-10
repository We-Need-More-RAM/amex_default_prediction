#####    IMPORTS    #####
import os

from kaggle.api.kaggle_api_extended import KaggleApi
#########################

#####    FUNCTIONS    #####

###########################

#####    MAIN SCRIPT    #####

invalid_file_given = True

while invalid_file_given:

    file_name = input('What is the name of the submission file? ')

    file_location = 'data/submission/' + file_name

    if os.path.isfile(file_location):
        
        invalid_file_given = False
        
api = KaggleApi()
api.authenticate()

api.competition_submit(file_name=file_location, 
                       message=f'submission of {file_name}', 
                       competition='amex-default-prediction'
                      )
#############################