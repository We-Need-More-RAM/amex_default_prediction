import pandas as pd
import scipy.stats
import numpy as np
from sklearn.model_selection import train_test_split

def acquire_amex(sample_size=199990):
    X_df = pd.read_csv('../../data/raw/train_data.csv', nrows = sample_size)
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
    cat_columns = ['B_30', 'B_31', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    X_df = pd.get_dummies(X_df, columns=cat_columns, drop_first=True)

    return X_df

def get_features(X_df):

    agg_df = X_df.groupby('customer_ID').agg(['last', 'std', 'min', 'max'])
    agg_df = collapse_columns(agg_df)

    missing_vals_df = get_null_count(X_df)
    zero_df = get_zeros(X_df)
    delta_df = get_delta_values(X_df)
    ema_df = get_ema(X_df)

    metrics_df = pd.concat([agg_df, missing_vals_df, zero_df, delta_df, ema_df],axis=1)

    metrics_df = get_pctb(X_df, metrics_df)
    metrics_df = get_range(X_df, metrics_df)
    metrics_df = get_cv(X_df, metrics_df)

    # drop the _min and _std columns. Those are captured in _range and _cv
    cols_to_drop = metrics_df.filter(regex='(_min|_std)$', axis=1).columns
    metrics_df = metrics_df.drop(columns=cols_to_drop)

    # drop those columns where > 90% of rows are missing values
    missing_counts_df = pd.DataFrame({'missing_count': metrics_df.isnull().sum(), 'missing_pct': metrics_df.isnull().sum()/len(metrics_df)})
    cols_to_drop = missing_counts_df[missing_counts_df.missing_pct > .90].index
    features_df = metrics_df.drop(columns=cols_to_drop)
    
    return features_df

def compute_entropy(features_df, ent):
    entropy_series = features_df.apply(ent)
    features_df = features_df[entropy_series[entropy_series > 1].index]
    return features_df

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



#################################
# Feature Engineering functions #
#################################

def collapse_columns(X_df):
    '''
    this function will collapse the multi-level index of the columns 
    that are generated after computing the first set of aggregates in 
    our groupby function in the agg_features function.
    '''
    # df = X_df.copy()
    if isinstance(X_df.columns, pd.MultiIndex):
        X_df.columns = X_df.columns.to_series().apply(lambda x: "_".join(x))
    return X_df

def get_null_count(X_df):
    '''
    this function will calculate the number of missing values for each feature. 
    it reaturns a dataframe with the columns: <column_name_orig>_nulls 
    '''
    missing_vals = X_df.groupby('customer_ID').agg(lambda x: x.isnull().sum())
    missing_vals.columns = [x + '_nulls' for x in missing_vals.columns]
    return missing_vals

def get_zeros(X_df):
    '''
    this function will calculate the number of zeros values for each feature. 
    it reaturns a dataframe with the columns: <column_name_orig>_zeros 
    '''
    zeros_df = X_df.groupby('customer_ID').agg(lambda x: (x == 0.0).sum())
    zeros_df.columns = [x + '_zeros' for x in zeros_df.columns]
    return zeros_df

# def get_cv(X_df):
#     '''
#     this function will compute the coefficient of variation for each feature. 
#     it reaturns a dataframe with the columns: <column_name_orig>_cv 
#     '''
#     cv_df = X_df.groupby('customer_ID').agg(lambda x: x.std()/x.mean())
#     cv_df.columns = [x + '_cv' for x in cv_df.columns]
#     return cv_df

def get_one_period_difference(X_df):
    '''
    This function computes the 2-period in values for each feature. 
    it returns a dataframe with the customer id set to the index. 
    the function is used in compute_delta_values() function
    '''
    delta_df = X_df.groupby('customer_ID').diff(periods=1)
    delta_df.index = X_df.customer_ID
    return delta_df

    
def get_delta_values(X_df):
    '''
    This function first gets the two-period difference in values for each feature and assigns that to a dataframe (delta).
    It generates a dataframe of the most recent 2-period difference (delta_value).
    Next, from the delta dataframe, it computes the number of negative deltas over the customer's history and 
    assigns that to a dataframe (neg_delta_count).
    Next, it uses the delta dataframe to compute the average delta over the customer's history and assigns that to 
    a dadtaframe (delta_mean).
    Finally, all of these dataframes are concatenated into a single dataframe, delta_df. 
    '''
    # first compute the 2 period delta and create a dataframe with those values
    delta_df = get_one_period_difference(X_df)
    delta_df.columns = [x + '_diff' for x in delta_df.columns]
    
    # Use the delta df to take the last value as the current delta
    delta_value = delta_df.groupby('customer_ID').last()
    
    # use the delta df to count the number of changes over customer history that were negative
    neg_delta_count = delta_df.groupby('customer_ID').agg(lambda x: (x < 0).sum())
    neg_delta_count.columns = [x + '_count' for x in delta_df.columns]
    
    # use the delta df to compute the rolling average of the delta values
    delta_mean = delta_df.groupby('customer_ID').transform(lambda x: x.rolling(window=6, 
                                                                       min_periods=1, 
                                                                       closed='left').mean())
    delta_mean.columns = [x + '_mean' for x in delta_df.columns]
    
    # take the last value, the current average of change
    delta_mean = delta_mean.groupby('customer_ID').last()
    
    # concatenate the dataframes with the computed values by concatenating columns along the customer index
    delta_df = pd.concat([delta_value, neg_delta_count, delta_mean], axis=1)
    return delta_df

def get_ema(X_df):
    '''
    This function will compute the exponential moving average, with an alpha of .8. 
    it returns a dataframe with the columns: <column_name_orig>_ema. 
    '''
    ema_df = X_df.groupby('customer_ID').transform(lambda x: x.ewm(alpha=.8, min_periods=1, adjust=True).mean().shift(periods=1))
    ema_df.columns = [x + '_ema' for x in ema_df.columns]
    ema_df.index = X_df.customer_ID
    ema_df = ema_df.groupby('customer_ID').last()
    return ema_df

def get_pctb(X_df, metrics_df):
    df_customer_indexed = X_df.set_index('customer_ID')
    pctb_series = pd.Series()

    # loop through original column names and for eacsh one, compute pctb
    k = 6
    for x in df_customer_indexed.columns:
        ubb = metrics_df[x + '_ema'] + k*metrics_df[x + '_std']
        lbb = metrics_df[(x + '_ema')] - k*metrics_df[x + '_std']
        pctb = (metrics_df[x + '_last'] - lbb) / (ubb - lbb)
        pctb_series = pd.concat([pctb_series, pctb], axis=1)
    
    pctb_df = pd.DataFrame(pctb_series)
    pctb_df = pctb_df.iloc[:,1:]
    pctb_df.columns = [x + '_pctb' for x in df_customer_indexed.columns]
    metrics_df = pd.concat([pctb_df, metrics_df], axis=1)
    return metrics_df

def get_range(X_df, metrics_df):
    df_customer_indexed = X_df.set_index('customer_ID')
    range_series = pd.Series()

    for x in df_customer_indexed.columns:
        range_val = metrics_df[x + '_max'] - metrics_df[x + '_min']
        range_series = pd.concat([range_series, range_val], axis=1)

    range_df = pd.DataFrame(range_series)
    range_df = range_df.iloc[:,1:]
    range_df.columns = [x + '_%b' for x in df_customer_indexed.columns]
    metrics_df = pd.concat([range_df, metrics_df], axis=1)
    return metrics_df

def get_cv(X_df, metrics_df):
    df_customer_indexed = X_df.set_index('customer_ID')
    cv_series = pd.Series()
    for x in df_customer_indexed.columns:
        cv = metrics_df[x + '_std']/metrics_df[x + '_ema']
        cv_series = pd.concat([cv_series, cv], axis=1)

    cv_df = pd.DataFrame(cv_series)
    cv_df = cv_df.iloc[:,1:]
    cv_df.columns = [x + '_cv' for x in df_customer_indexed.columns]
    metrics_df = pd.concat([cv_df, metrics_df], axis=1)
    return metrics_df

def ent(data):
    """Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()           # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy


