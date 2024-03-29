import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

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


def age_calculator(df):
    
    def datetimer(string):
        return dt.datetime.strptime(string, '%Y-%m-%d')
    
    df['S_2'] = df['S_2'].apply(datetimer)
    
    min_dates = pd.DataFrame(df.groupby('customer_ID').S_2.min())
                             
    min_dates.rename(columns={'S_2': 'min_dates'}, inplace=True)
                             
    df = df.merge(min_dates, how='left', on='customer_ID')
                             
    df['age'] = (df['S_2'] - df['min_dates']).dt.days
                             
    df.drop(columns=['S_2', 'min_dates'], inplace=True)
                             
    return df


def initial_prep(df):
    
    df.drop(columns=['S_2'], inplace=True)
    
    return df
                             

def train_null_counter(df):
    
    df_count = df.groupby('customer_ID').agg(lambda x: x.isnull().sum())
    
    df_count.columns = [x + '_nulls' for x in df_count.columns]
    
    ss = StandardScaler()
    
    df_count[df_count.columns] = ss.fit_transform(df_count[df_count.columns])
    
    return df_count, ss
    

def handle_categories(df):
    
    categorical_columns = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    
    dummy_df = pd.get_dummies(df[categorical_columns].astype('category'), dummy_na=True)
    
    id_df = pd.DataFrame(df['customer_ID'])
    
    dummy_df = pd.concat([id_df, dummy_df], axis=1)
    
    dummy_last = dummy_df.groupby('customer_ID').agg('last')
    
    return dummy_last, categorical_columns


def cap_numerical_columns(df, cat_cols):
    
    numerical_cols = list(df.drop(columns=cat_cols))
    
    numerical_cols.remove('customer_ID')
    
    for col in numerical_cols:
        df[col].clip(lower=-3, upper=3, inplace=True)
        
    return df, numerical_cols


def impute_numerical_nulls(df, num_cols, k=5):
    
    for col in num_cols:
        df[col].fillna(k, inplace=True)
        
    return df    
                       

def aggregate_features(df, num_cols):
    
    new_df = pd.DataFrame({'customer_ID': df['customer_ID'].unique()})
    
    for col in num_cols:
        mini = df.groupby('customer_ID')[col].min()
        maxi = df.groupby('customer_ID')[col].max()
        med = df.groupby('customer_ID')[col].median()
        std = df.groupby('customer_ID')[col].std()
        first = df.groupby('customer_ID')[col].first()
        last = df.groupby('customer_ID')[col].last()
        change = last - first
    
        stats = pd.DataFrame({f'{col}_min': mini, f'{col}_max': maxi,
                              f'{col}_median': med, f'{col}_std': std,
                              f'{col}_last': last, f'{col}_change': change})
    
        stats.reset_index(drop=True, inplace=True)
    
        new_df = pd.concat([new_df, stats], axis=1)
        
    new_df.set_index('customer_ID', inplace=True)
    
    new_df.fillna(0, inplace=True)
    
    return new_df


def concat_dataframes(agg_df, dummy_df, null_df):
    
    df = pd.concat([agg_df, dummy_df], axis=1)
    
    df = pd.concat([df, null_df], axis=1)
    
    return df


def valid_null_counter(df, scaler):
    
    df_count = df.groupby('customer_ID').agg(lambda x: x.isnull().sum())
    
    df_count.columns = [x + '_nulls' for x in df_count.columns]
    
    df_count[df_count.columns] = scaler.transform(df_count[df_count.columns])
    
    return df_count


def prep_data(df):
    
    print(f'Beginning Prep at {datetime.now()}')
    
    df = initial_prep(df)
    
    null_df, ss = train_null_counter(df)
    
    dummy_df, cat_columns = handle_categories(df)
    
    df, num_columns = cap_numerical_columns(df, cat_columns)
    
    df = impute_numerical_nulls(df, num_columns)
    
    agg_features = aggregate_features(df, num_columns)
    
    final_df = concat_dataframes(agg_features, dummy_df, null_df)
    
    print(f'Finished Prep at {datetime.now()}')
    
    return final_df
    