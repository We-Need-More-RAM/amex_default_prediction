import pandas as pd
import datetime as dt

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
                             

def value_haver(train, col_list):
    
    df = pd.DataFrame(index=train['customer_ID'].unique())
    
    for col in col_list:
        df[col + '_has_value'] = train.groupby('customer_ID')[col].sum()
        df[col + '_has_value'].where(df[col + '_has_value'] == 0, other=1, inplace=True)
                             
    return df
                             
                             
def changefinder(df, train, col_list):
                             
    for col in col_list:
        df[col + '_change'] = train[train.groupby('customer_ID')['age'].transform(max) == train['age']][col].to_numpy() -                                               train[train.groupby('customer_ID')['age'].transform(min) == train['age']][col].to_numpy()
                             
    return df                         