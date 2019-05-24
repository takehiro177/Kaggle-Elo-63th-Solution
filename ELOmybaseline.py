
#%%
import datetime
import gc
import sys
import time
import warnings

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error,log_loss
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

new_transactions = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/new_merchant_transactions.csv', parse_dates=['purchase_date']))
historical_transactions = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/historical_transactions.csv', parse_dates=['purchase_date']))

new_transactions['purchase_amount'] = (new_transactions['purchase_amount']+0.761920783094309)*66.5423961054881
historical_transactions['purchase_amount'] = (historical_transactions['purchase_amount']+0.761920783094309)*66.5423961054881


def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)


def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df

# Process transaction data
#impute
historical_transactions['category_2'] = historical_transactions['category_2'].fillna(1.0,inplace=True)
historical_transactions['category_3'] = historical_transactions['category_3'].fillna('A',inplace=True)

new_transactions['category_2'] = new_transactions['category_2'].fillna(1.0,inplace=True)
new_transactions['category_3'] = new_transactions['category_3'].fillna('A',inplace=True)

historical_transactions['category_3'] = historical_transactions['category_3'].map({'A':0, 'B':1, 'C':2})
new_transactions['category_3'] = new_transactions['category_3'].map({'A':0, 'B':1, 'C':2})
#end

#----------holidays processing
historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])

agg_func = {
        'mean': ['mean'],
    }

historical_transactions['year'] = historical_transactions['purchase_date'].dt.year
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['dayofweek'] = historical_transactions['purchase_date'].dt.dayofweek
historical_transactions['weekend'] = (historical_transactions.purchase_date.dt.weekday >=5).astype(int)
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour 
historical_transactions['quarter'] = historical_transactions['purchase_date'].dt.quarter
historical_transactions['is_month_start'] = historical_transactions['purchase_date'].dt.is_month_start
historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
historical_transactions['month_diff'] += historical_transactions['month_lag']
for col in ['category_2','category_3']:
    historical_transactions[col+'_mean'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('mean')
    historical_transactions[col+'_max'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('max')
    historical_transactions[col+'_min'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('min')
    historical_transactions[col+'_var'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('var')
    agg_func[col+'_mean'] = ['mean']

new_transactions['year'] = new_transactions['purchase_date'].dt.year
new_transactions['weekofyear'] = new_transactions['purchase_date'].dt.weekofyear
new_transactions['month'] = new_transactions['purchase_date'].dt.month
new_transactions['dayofweek'] = new_transactions['purchase_date'].dt.dayofweek
new_transactions['weekend'] = (new_transactions.purchase_date.dt.weekday >=5).astype(int)
new_transactions['hour'] = new_transactions['purchase_date'].dt.hour 
new_transactions['quarter'] = new_transactions['purchase_date'].dt.quarter
new_transactions['is_month_start'] = new_transactions['purchase_date'].dt.is_month_start
new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']
for col in ['category_2','category_3']:
    new_transactions[col+'_mean'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('mean')
    new_transactions[col+'_max'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('max')
    new_transactions[col+'_min'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('min')
    new_transactions[col+'_var'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('var')
    agg_func[col+'_mean'] = ['mean']

gc.collect()


# New Features with Key Shopping times considered in the dataset. if the purchase has been made within 60 days, it is considered as an influence
#Christmas : December 25 2017
historical_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#fathers day: August 13 2017
historical_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Childrens day: October 12 2017
historical_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Black Friday : 24th November 2017
historical_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Mothers Day: May 14 2017
historical_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-05-14') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-05-14') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Valentines Day
historical_transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)


gc.collect()


#---------- Id detail mode
mode_list = historical_transactions.groupby('card_id')['city_id','merchant_category_id','state_id','subsector_id','month_lag'].apply(lambda x: x.mode().iloc[0])
mode_list.columns = ['mode_' + c if c != 'card_id' else c for c in mode_list.columns]
historical_transactions = pd.merge(historical_transactions,mode_list,on='card_id',how='left')

mode_list = new_transactions.groupby('card_id')['city_id','merchant_category_id','state_id','subsector_id','month_lag'].apply(lambda x: x.mode().iloc[0])
mode_list.columns = ['mode_' + c if c != 'card_id' else c for c in mode_list.columns]
new_transactions = pd.merge(new_transactions,mode_list,on='card_id',how='left')
del mode_list;gc.collect()

#-----------Id detail merchant_id repeat

'''
max_duplicate = new_transactions.groupby(['card_id','merchant_id']).size().max(level='card_id')
mean_duplicate = new_transactions.groupby(['card_id','merchant_id']).size().mean(level='card_id')
max_duplicate = max_duplicate.to_frame()
mean_duplicate = mean_duplicate.to_frame()
new_transactions = pd.merge(new_transactions,max_duplicate,on='card_id',how='left')
new_transactions = pd.merge(new_transactions,mean_duplicate,on='card_id',how='left')
new_transactions = new_transactions.rename(columns={'0_x':'max_duplicate'})
new_transactions = new_transactions.rename(columns={'0_y':'mean_duplicate'})

del max_duplicate,mean_duplicate;gc.collect()
'''

#-------------------------

historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month

agg_fun = {'authorized_flag': ['mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['auth_mean_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
non_authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]

hist_weekend_transactions = historical_transactions[historical_transactions['weekend'] == 1]
new_weekend_transactions = new_transactions[new_transactions['weekend'] == 1]

gc.collect()


def aggregate_transactions(history):
    
    
    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': ['min', 'max'],
        'month_lag': ['min', 'max','std','var'],
        #
        'month_diff' : ['mean', 'min', 'max', 'var'],
        'weekend' : ['sum', 'mean'],
        'card_id' : ['size'],
        'month': ['nunique'],
        'hour': ['nunique'],
        'quarter':['nunique'],
        'weekofyear': ['nunique'],                
        'dayofweek': ['nunique'],
        'Christmas_Day_2017':['mean','max'],
        'fathers_day_2017':['mean','max'],
        'Children_day_2017':['mean','max'],        
        'Black_Friday_2017':['mean','max'],
        'Valentine_day_2017':['mean'],
        'Mothers_Day_2017':['mean'],
        #
        'mode_city_id':['mean'],
        'mode_merchant_category_id':['mean'],
        'mode_state_id':['mean'],
        'mode_subsector_id':['mean'],
        'mode_month_lag':['mean'],


    }

    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

def aggregate_transactions_weekend(history):
    
    
    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        #
        'mode_city_id':['mean'],
        'mode_merchant_category_id':['mean'],
        'mode_state_id':['mean'],
        'mode_subsector_id':['mean'],
        'mode_month_lag':['mean'],

    }

    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

gc.collect()

train = reduce_mem_usage(read_data('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/train.csv'))
test = reduce_mem_usage(read_data('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/test.csv'))

print('processing train and test merge')

train = pd.merge(train, auth_mean, on='card_id', how='left')
test = pd.merge(test, auth_mean, on='card_id', how='left')
del auth_mean;gc.collect()

history = aggregate_transactions(non_authorized_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
del non_authorized_transactions;gc.collect()

train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
del history;gc.collect()

authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
del authorized_transactions;gc.collect()

train = pd.merge(train, authorized, on='card_id', how='left')
test = pd.merge(test, authorized, on='card_id', how='left')
del authorized;gc.collect()

hist_weekend =aggregate_transactions_weekend(hist_weekend_transactions)
hist_weekend.columns = ['hist_weekend_' + c if c != 'card_id' else c for c in hist_weekend.columns]
del hist_weekend_transactions;gc.collect()

train = pd.merge(train, hist_weekend, on='card_id', how='left')
test = pd.merge(test, hist_weekend, on='card_id', how='left')
del hist_weekend;gc.collect()

new_weekend =aggregate_transactions_weekend(new_weekend_transactions)
new_weekend.columns = ['new_weekend_' + c if c != 'card_id' else c for c in new_weekend.columns]
del new_weekend_transactions;gc.collect()

train = pd.merge(train, new_weekend, on='card_id', how='left')
test = pd.merge(test, new_weekend, on='card_id', how='left')
del new_weekend;gc.collect()

new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')
del new;gc.collect()

print('processing hist merchant id')
#--------unique merchant_id  ------------------hist
m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_57df19bf28'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_57df19bf28'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_19171c737a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_19171c737a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_8fadd601d2'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_8fadd601d2'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9e84cda3b1'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9e84cda3b1'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_b794b9d9e8'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_b794b9d9e8'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_ec24d672a3'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_ec24d672a3'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_5a0a412718'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_5a0a412718'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_490f186c5a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_490f186c5a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_77e2942cd8'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_77e2942cd8'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_fc7d7969c3'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_fc7d7969c3'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_0a00fa9e8a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_0a00fa9e8a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_1f4773aa76'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_1f4773aa76'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_5ba019a379'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_5ba019a379'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_ae9fe1605a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_ae9fe1605a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_48257bb851'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_48257bb851'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_1d8085cf5d'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_1d8085cf5d'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_820c7b73c8'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_820c7b73c8'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_1ceca881f0'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_1ceca881f0'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_00a6ca8a8a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'hist_M_ID_00a6ca8a8a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_e5374dabc0'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'hist_M_ID_e5374dabc0'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9139332ccc'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'hist_M_ID_9139332ccc'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_5d4027918d'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_5d4027918d'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9fa00da7b2'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9fa00da7b2'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_d7f0a89a87'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_d7f0a89a87'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_daeb0fe461'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_daeb0fe461'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_077bbb4469'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_077bbb4469'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_f28259cb0a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_f28259cb0a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_0a767b8200'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_0a767b8200'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_fee47269cb'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_fee47269cb'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_c240e33141'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_c240e33141'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_a39e6f1119'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_a39e6f1119'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_c8911208f2'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_c8911208f2'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9a06a8cf31'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9a06a8cf31'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_08fdba20dc'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_08fdba20dc'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_a483a17d19'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_a483a17d19'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_aed77085ce'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_aed77085ce'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_25d0d2501a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_25d0d2501a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_69f024d01a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_69f024d01a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')



del historical_transactions
#-----------------------mode by rank of merchant count

''''
print('processing mode by rank of hist merchant count')
count_merchant_rank = historical_transactions['merchant_id'].value_counts()

count_merchant_rank = count_merchant_rank.to_dict()
count_merchant_rank = {i:count_merchant_rank[i] for i in count_merchant_rank if count_merchant_rank[i] != 1}

step = 1
for key in count_merchant_rank:
    count_merchant_rank[key] = step
    step += 1
historical_transactions['merchant_id'] = historical_transactions['merchant_id'].fillna(-1)
historical_transactions['merchant_id'] = historical_transactions['merchant_id'].map(count_merchant_rank).fillna(step)
m = historical_transactions.groupby('card_id')['merchant_id'].apply(lambda x: x.mode().iloc[0])
m = m.to_frame()
m = m.rename(columns={'merchant_id':'mode_hist_merchant_id_rank'})
del historical_transactions,step,key,count_merchant_rank;gc.collect()

train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')
del m;gc.collect()

print('processing mode by rank of new merchant count')
count_merchant_rank = new_transactions['merchant_id'].value_counts()

count_merchant_rank = count_merchant_rank.to_dict()
count_merchant_rank = {i:count_merchant_rank[i] for i in count_merchant_rank if count_merchant_rank[i] != 1}

step = 1
for key in count_merchant_rank:
    count_merchant_rank[key] = step
    step += 1
new_transactions['merchant_id_mode'] = new_transactions['merchant_id'].fillna(-1)
new_transactions['merchant_id_mode'] = new_transactions['merchant_id_mode'].map(count_merchant_rank).fillna(step)
m = new_transactions.groupby('card_id')['merchant_id_mode'].apply(lambda x: x.mode().iloc[0])
m = m.to_frame()
m = m.rename(columns={'merchant_id_mode':'mode_new_merchant_id_rank'})
del step,key,count_merchant_rank;gc.collect()

train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')
del m;gc.collect()
'''

#---------------unique merchant id ------------new
print('processing new merchant id')
m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_6f274b9340'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_6f274b9340'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_445742726b'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_445742726b'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_4e461f7e14'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_4e461f7e14'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_e5374dabc0'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_e5374dabc0'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_3111c6df35'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_3111c6df35'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_50f575c681'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_50f575c681'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_00a6ca8a8a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_00a6ca8a8a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_cd2c0b07e9'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_cd2c0b07e9'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9139332ccc'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9139332ccc'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


del m, new_transactions;gc.collect()


# CV sampling uniform distribution
train['rounded_target'] = train['target'].round(0)
train = train.sort_values('rounded_target').reset_index(drop=True)
vc = train['rounded_target'].value_counts()
vc = dict(sorted(vc.items()))
df = pd.DataFrame()
train['indexcol'],i = 0,1
for k,v in vc.items():
    step = train.shape[0]/v
    indent = train.shape[0]/(v+1)
    df2 = train[train['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df = pd.concat([df2,df])
    i+=1
train = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
del train['indexcol'], train['rounded_target']


train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1
outliers = train['outliers']
target = train['target']
#del train['target']
gc.collect()


# Now extract the month, year, day, weekday

train["month"] = train["first_active_month"].dt.month
train["year"] = train["first_active_month"].dt.year
train['week'] = train["first_active_month"].dt.weekofyear
train['dayofweek'] = train['first_active_month'].dt.dayofweek
train['days'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days

test["month"] = test["first_active_month"].dt.month
test["year"] = test["first_active_month"].dt.year
test['week'] = test["first_active_month"].dt.weekofyear
test['dayofweek'] = test['first_active_month'].dt.dayofweek
test['days'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days


#Feature Engineering - Adding new features inspired by Chau's first kernel
train['hist_purchase_date_max'] = pd.to_datetime(train['hist_purchase_date_max'])
train['hist_purchase_date_min'] = pd.to_datetime(train['hist_purchase_date_min'])
train['hist_purchase_date_diff'] = (train['hist_purchase_date_max'] - train['hist_purchase_date_min']).dt.days
train['hist_purchase_date_average'] = train['hist_purchase_date_diff']/train['hist_card_id_size']
train['hist_purchase_date_uptonow'] = (datetime.datetime.today() - train['hist_purchase_date_max']).dt.days
train['hist_purchase_date_uptomin'] = (datetime.datetime.today() - train['hist_purchase_date_min']).dt.days
train['hist_first_buy'] = (train['hist_purchase_date_min'] - train['first_active_month']).dt.days
for feature in ['hist_purchase_date_max','hist_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

test['hist_purchase_date_max'] = pd.to_datetime(test['hist_purchase_date_max'])
test['hist_purchase_date_min'] = pd.to_datetime(test['hist_purchase_date_min'])
test['hist_purchase_date_diff'] = (test['hist_purchase_date_max'] - test['hist_purchase_date_min']).dt.days
test['hist_purchase_date_average'] = test['hist_purchase_date_diff']/test['hist_card_id_size']
test['hist_purchase_date_uptonow'] = (datetime.datetime.today() - test['hist_purchase_date_max']).dt.days
test['hist_purchase_date_uptomin'] = (datetime.datetime.today() - test['hist_purchase_date_min']).dt.days
test['hist_first_buy'] = (test['hist_purchase_date_min'] - test['first_active_month']).dt.days
for feature in ['hist_purchase_date_max','hist_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

#Feature Engineering - Adding new features inspired by Chau's first kernel
train['auth_purchase_date_max'] = pd.to_datetime(train['auth_purchase_date_max'])
train['auth_purchase_date_min'] = pd.to_datetime(train['auth_purchase_date_min'])
train['auth_purchase_date_diff'] = (train['auth_purchase_date_max'] - train['auth_purchase_date_min']).dt.days
train['auth_purchase_date_average'] = train['auth_purchase_date_diff']/train['auth_card_id_size']
train['auth_purchase_date_uptonow'] = (datetime.datetime.today() - train['auth_purchase_date_max']).dt.days
train['auth_purchase_date_uptomin'] = (datetime.datetime.today() - train['auth_purchase_date_min']).dt.days
train['auth_first_buy'] = (train['auth_purchase_date_min'] - train['first_active_month']).dt.days
for feature in ['auth_purchase_date_max','auth_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

test['auth_purchase_date_max'] = pd.to_datetime(test['auth_purchase_date_max'])
test['auth_purchase_date_min'] = pd.to_datetime(test['auth_purchase_date_min'])
test['auth_purchase_date_diff'] = (test['auth_purchase_date_max'] - test['auth_purchase_date_min']).dt.days
test['auth_purchase_date_average'] = test['auth_purchase_date_diff']/test['auth_card_id_size']
test['auth_purchase_date_uptonow'] = (datetime.datetime.today() - test['auth_purchase_date_max']).dt.days
test['auth_purchase_date_uptomin'] = (datetime.datetime.today() - test['auth_purchase_date_min']).dt.days
test['auth_first_buy'] = (test['auth_purchase_date_min'] - test['first_active_month']).dt.days
for feature in ['auth_purchase_date_max','auth_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

#Feature Engineering - Adding new features inspired by Chau's first kernel
train['new_purchase_date_max'] = pd.to_datetime(train['new_purchase_date_max'])
train['new_purchase_date_min'] = pd.to_datetime(train['new_purchase_date_min'])
train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days
train['new_purchase_date_average'] = train['new_purchase_date_diff']/train['new_card_id_size']
train['new_purchase_date_uptonow'] = (datetime.datetime.today() - train['new_purchase_date_max']).dt.days
train['new_purchase_date_uptomin'] = (datetime.datetime.today() - train['new_purchase_date_min']).dt.days
train['new_first_buy'] = (train['new_purchase_date_min'] - train['first_active_month']).dt.days
for feature in ['new_purchase_date_max','new_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

test['new_purchase_date_max'] = pd.to_datetime(test['new_purchase_date_max'])
test['new_purchase_date_min'] = pd.to_datetime(test['new_purchase_date_min'])
test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days
test['new_purchase_date_average'] = test['new_purchase_date_diff']/test['new_card_id_size']
test['new_purchase_date_uptonow'] = (datetime.datetime.today() - test['new_purchase_date_max']).dt.days
test['new_purchase_date_uptomin'] = (datetime.datetime.today() - test['new_purchase_date_min']).dt.days
test['new_first_buy'] = (test['new_purchase_date_min'] - test['first_active_month']).dt.days
for feature in ['new_purchase_date_max','new_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

#Feature new vs auth
train['diff_purchase_date_diff'] =  train['auth_purchase_date_diff'] - train['new_purchase_date_diff'] 
train['diff_purchase_date_average'] = train['auth_purchase_date_average'] - train['new_purchase_date_average']
train['hist_00a6ca8a8a_ratio'] = train['hist_M_ID_00a6ca8a8a']/(train['hist_transactions_count']+train['auth_transactions_count'])
train['new_00a6ca8a8a_ratio'] = train['M_ID_00a6ca8a8a']/train['new_transactions_count']

test['diff_purchase_date_diff'] =  test['auth_purchase_date_diff'] - test['new_purchase_date_diff']
test['diff_purchase_date_average'] = test['auth_purchase_date_average'] - test['new_purchase_date_average']
test['hist_00a6ca8a8a_ratio'] = test['hist_M_ID_00a6ca8a8a']/(test['hist_transactions_count']+test['auth_transactions_count'])
test['new_00a6ca8a8a_ratio'] = test['M_ID_00a6ca8a8a']/test['new_transactions_count']

#Feature for combination
train['category1_auth_ratio'] = train['auth_category_1_sum']/train['auth_transactions_count']
train['category_1_new_ratio'] = train['new_category_1_sum']/train['new_transactions_count']
train['date_average_new_auth_ratio'] = train['auth_purchase_date_average']/train['new_purchase_date_average']
train['childday_ratio'] = train['auth_Children_day_2017_mean']/train['new_Children_day_2017_mean']
train['blackday_ratio'] = train['auth_Black_Friday_2017_mean']/train['new_Black_Friday_2017_mean']
train['fatherday_ratio'] = train['auth_fathers_day_2017_mean']/train['new_fathers_day_2017_mean']
train['christmasday_ratio'] = train['auth_Christmas_Day_2017_mean']/train['new_Christmas_Day_2017_mean']
train['date_uptonow_diff_auth_new'] = train['auth_purchase_date_uptonow'] - train['new_purchase_date_uptonow']

test['category1_auth_ratio'] = test['auth_category_1_sum']/test['auth_transactions_count']
test['category_1_new_ratio'] = test['new_category_1_sum']/test['new_transactions_count']
test['date_average_new_auth_ratio'] = test['auth_purchase_date_average']/test['new_purchase_date_average']
test['childday_ratio'] = test['auth_Children_day_2017_mean']/test['new_Children_day_2017_mean']
test['blackday_ratio'] = test['auth_Black_Friday_2017_mean']/test['new_Black_Friday_2017_mean']
test['fatherday_ratio'] = test['auth_fathers_day_2017_mean']/test['new_fathers_day_2017_mean']
test['christmasday_ratio'] = test['auth_Christmas_Day_2017_mean']/test['new_Christmas_Day_2017_mean']
test['date_uptonow_diff_auth_new'] = test['auth_purchase_date_uptonow'] - test['new_purchase_date_uptonow']

#Feature weekend for combination

train['category1_hist_weekend_ratio'] = train['hist_weekend_category_1_sum']/train['hist_weekend_transactions_count']
train['category_1_new_weekend_ratio'] = train['new_weekend_category_1_sum']/train['new_weekend_transactions_count']
test['category1_hist_weekend_ratio'] = test['hist_weekend_category_1_sum']/test['hist_weekend_transactions_count']
test['category_1_new_weekend_ratio'] = test['new_weekend_category_1_sum']/test['new_weekend_transactions_count']

train = train.fillna(0)
test = test.fillna(0)

train.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/train_c.csv')
test.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/test_c.csv')

features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
train = train[features]
test = test[features]

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
gc.collect()

#------------------------------------Feature Selection with Null Importances


def get_feature_importances(train,target, shuffle, seed=None):

    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = target
    if shuffle:
        # Here you could as well use a binomial distribution
        y = pd.DataFrame(y).sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False, silent=True)
    lgb_params = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 8,
         }
    
    
    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=3500, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(train[features]))
    
    return imp_df

def get_feature_importances2(train,target, shuffle, seed=None):

    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = target
    if shuffle:
        # Here you could as well use a binomial distribution
        y = pd.DataFrame(y).sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False, silent=True)
    params ={
            'task': 'train',
            'boosting': 'goss',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9,
            'max_depth': 7,
            'top_rate': 0.9,
            'num_leaves': 63,
            'min_child_weight': 40.,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.,
            'colsample_bytree': 0.5,
            'min_split_gain': 9,
            'reg_lambda': 8.,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed':int(2**7),
            'bagging_seed':int(2**7),
            'drop_seed':int(2**7)
            }      
    
    
    # Fit the model
    clf = lgb.train(params=params, train_set=dtrain, num_boost_round=3500, categorical_feature=categorical_feats)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(train[features]))
    
    return imp_df

actual_imp_df = get_feature_importances(train=train,target=target,shuffle=False)

null_imp_df = pd.DataFrame()
nb_runs = 80
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(train=train,target=target,shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='')



correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))

corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
gc.collect()

corr_scores_df.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestmodel_stacking/feature_correlation_main_lgb1.csv', index=False)
#------------------------------------------
actual_imp_df = get_feature_importances2(train=train,target=target,shuffle=False)

null_imp_df = pd.DataFrame()
nb_runs = 80
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(train=train,target=target,shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='')



correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))

corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
gc.collect()

corr_scores_df.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestmodel_stacking/feature_correlation_main_lgb2.csv', index=False)


#%%

def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
    # Fit LightGBM 
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 8,
         }

    params ={
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'top_rate': 0.9064148448434349,
            'num_leaves': 63,
            'min_child_weight': 41.9612869171337,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'reg_lambda': 8.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed':int(2**7),
            'bagging_seed':int(2**7),
            'drop_seed':int(2**7)
            }

    
    # Fit the model
    hist = lgb.cv(
        params=lgb_params, 
        train_set=dtrain, 
        num_boost_round= 3000,
        categorical_feature=cat_feats,
        nfold=7,
        shuffle=False,
        early_stopping_rounds=400,
        verbose_eval=0,
        seed=17,
        stratified=False
    )
    # Return the last mean / std values 
    return hist['rmse-mean'][-1], hist['rmse-stdv'][-1]

split_feats = []
split_cat_feats = []
gain_feats = []
gain_cat_feats = []

for threshold in [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]:

    for _f in corr_scores_df.itertuples():
        if _f[2] >= threshold:
            split_feats.append(_f[1])

    for  _f in corr_scores_df.itertuples():
        if (_f[2] >= threshold) & (_f[1] in categorical_feats):
            split_cat_feats.append(_f[1])

    for _f in corr_scores_df.itertuples():
       if _f[3] >= threshold:
           gain_feats.append(_f[1])
    
    for _f  in corr_scores_df.itertuples():
        if (_f[3] >= threshold) & (_f[1] in categorical_feats):
            gain_cat_feats.append(_f[1])
                                                                                             
    print('Results for threshold %3d' % threshold)
    split_results = score_feature_selection(df=train, train_features=split_feats, cat_feats=split_cat_feats, target=target)
    print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
    gain_results = score_feature_selection(df=train, train_features=gain_feats, cat_feats=gain_cat_feats, target=target)
    print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
    split_feats = []
    split_cat_feats = []
    gain_feats = []
    gain_cat_feats = []


#-------------------------------------
# feature to try to remove done: best threshould is 
#-------------------------------------


#%%

corr_scores_df = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestmodel_stacking/feature_correlation_main_lgb2.csv')


threshold = 60
featuresFM = []
categorical_feats_split = []
for _f in corr_scores_df.itertuples():
    if _f[2] >= threshold:
        featuresFM.append(_f[1])

for  _f in corr_scores_df.itertuples():
    if (_f[2] >= threshold) & (_f[1] in categorical_feats):
        categorical_feats_split.append(_f[1])

print(len(featuresFM))
train = train[featuresFM]
test = test[featuresFM]

del corr_scores_df;gc.collect()

#%%

features = train.columns.values

param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 7,
         "verbosity": -1}

params ={
            'task': 'train',
            'boosting': 'goss',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9,
            'max_depth': 7,
            'top_rate': 0.9,
            'num_leaves': 63,
            'min_child_weight': 40.,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.,
            'colsample_bytree': 0.5,
            'min_split_gain': 9,
            'reg_lambda': 8.,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed':int(2**7),
            'bagging_seed':int(2**7),
            'drop_seed':int(2**7)
            }      
    

folds = KFold(n_splits=7, shuffle=False, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 400)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits


print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))


sample_submission = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestline_submission_main_fillnan.csv', index=False)




#%%


train.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/train_c.csv')
test.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/test_c.csv')

