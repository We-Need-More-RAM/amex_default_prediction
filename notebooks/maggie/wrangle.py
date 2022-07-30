import pandas as pd

from sklearn.model_selection import train_test_split

def acquire_amex(sample_size=200000):
    X_df = pd.read_csv('../../data/raw/train_data.csv', nrows=sample_size)
    y_df = pd.read_csv('../../data/raw/train_labels.csv')
    return X_df, y_df

def clean_amex(X_df):
    '''
    This function takes in features dataframe (X_df)
    and prepares in the following way:
    convert S_2 to datetime, create dummy variables for columns D_63, D_64 and B_31
    '''
    # convert S_2 to datetime
    X_df['S_2'] = pd.to_datetime(X_df.S_2)
    
    # For columns D_63, D_64 and B_31, we will want to create dummy variables. 
    X_df = pd.get_dummies(X_df, columns=['D_63', 'D_64', 'B_31'], drop_first=True)

    return X_df

def split_amex(X_df, y_df, train_size=.5, test_size=.5):
    '''
    This function takes in, as arguments, the features dataframe (X_df), the labels dataframe (y_df), 
    train_size proportion, and final test size proportion, and returns the following dataframes:
    X_train, y_train, X_validate, y_validate, X_test, y_test
    '''
    # split on y
    y_train, y_validate_test = train_test_split(y_df, train_size=0.20, random_state=13)
    y_validate, y_test = train_test_split(y_validate_test, test_size=0.49, random_state=13)
    # use customer ID's in y to split on X
    X_train = X_df[X_df.customer_ID.isin(y_train.customer_ID.unique())]
    X_validate = X_df[X_df.customer_ID.isin(y_validate.customer_ID.unique())]
    X_test = X_df[X_df.customer_ID.isin(y_test.customer_ID.unique())]
    return X_train, y_train, X_validate, y_validate, X_test, y_test
