import os

import pandas as pd

from sklearn.model_selection import train_test_split


# Write supporting functions here
def check_for_prepared_files():
    
    necessary_files = ['train_labels.csv',
                       'test_labels.csv',
                       'val_labels.csv',
                       'train_data.csv',
                       'val_data.csv',
                       'test_data.csv',
                      ]
    
    for file in necessary_files:
        
        if not os.path.isfile(f'data/prepared/{file}'):
            
            return False
    
    print('Prepare: Prepared files already exist! Skipping prep!')
    return True

def split_data():
    X_df = pd.read_csv('data/raw/train_data.csv')
    y_df = pd.read_csv('data/raw/train_labels.csv')
    
    train_labels, val_test = train_test_split(y_df, train_size=0.085, random_state=13)
    val_labels, test_labels = train_test_split(val_test, test_size=0.49, random_state=13)

    print('    Train Labels: %d rows, %d cols' % train_labels.shape)
    print('    Validate Labels: %d rows, %d cols' % val_labels.shape)
    print('    Test Labels: %d rows, %d cols' % test_labels.shape)
    
    train_data = X_df[X_df.customer_ID.isin(train_labels.customer_ID.unique())]
    val_data = X_df[X_df.customer_ID.isin(val_labels.customer_ID.unique())]
    test_data = X_df[X_df.customer_ID.isin(test_labels.customer_ID.unique())]

    print('    Train Data: %d rows, %d cols' % train_data.shape)
    print('    Validate Data: %d rows, %d cols' % val_data.shape)
    print('    Test Data: %d rows, %d cols' % test_data.shape)
    
    return train_labels, val_labels, test_labels, train_data, val_data, test_data

def store_prepared_data(train_labels, val_labels, test_labels, train_data, val_data, test_data):
    
    train_labels.to_csv('data/prepared/train_labels.csv', index=False)
    val_labels.to_csv('data/prepared/val_labels.csv', index=False)
    test_labels.to_csv('data/prepared/test_labels.csv', index=False)

    train_data.to_csv('data/prepared/train_data.csv', index=False)
    val_data.to_csv('data/prepared/val_data.csv', index=False)
    test_data.to_csv('data/prepared/test_data.csv', index=False)
    

def run():
    print('Prepare: Beginning prep script...')
    
    files_exist = check_for_prepared_files()
    
    if not files_exist:
    
        print("Prepare: Splitting acquired data...")
        train_labels, val_labels, test_labels, train_data, val_data, test_data = split_data()
        print("Prepare: Splitting acquired complete!")


        print("Prepare: Cleaning acquired data...")

        print("Prepare: Cleaning acquired data complete!")


        print("Prepare: Writing prepared datasets...")
        store_prepared_data(train_labels, val_labels, test_labels, train_data, val_data, test_data)
        print("Prepare: Writing prepared datasets complete!")

    print("Prepare: Completed!")
