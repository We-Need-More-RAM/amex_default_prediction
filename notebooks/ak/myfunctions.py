import pandas as pd
import datetime as dtq
from sklearn.preprocessing import StandardScaler

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


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
                             

def null_counter(df):
    
    df_count = df.groupby('customer_ID').agg(lambda x: x.isnull().sum())
    
    df_count.columns = [x + '_nulls' for x in df_count.columns]
    
    ss = StandardScaler()
    
    df_count[df_count.columns] = ss.fit_transform(df_count[df_count.columns])
    
    return df_count, ss
    

def handle_categories(df):
    
    categorical_columns = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    
    dummy_df = pd.get_dummies(df[categorical_columns], dummy_na=True)
    
    dummy_last = dummy_df.groupby('customer_ID').agg('last')
    
    return dummy_last


def cap_numerical_columns(df):
    
    numerical_columns = 