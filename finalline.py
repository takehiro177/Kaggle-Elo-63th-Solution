
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
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import precision_score

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

#%%
train = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/train_c.csv'))
test = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/test_c.csv'))

target = train['target']
features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
train = train[features]
test = test[features]

corr_scores_df = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/feature_correlation_main_lgb1.csv')

#%%
threshold = 70
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

threshold = [-27,-26,-25,-24,-23]
for thre in threshold:
    oof = [-33.218750 if x < thre else x for x in oof]
    print("CV score for threshould {}: {:<8.5f}".format(thre,mean_squared_error(oof, target)**0.5))

#%%
sample_submission = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestline_submission_main.csv', index=False)

#%%
#-----------------trainning without outlier----------------------------------------

train = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/train_c.csv'))
test = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/test_c.csv'))

train = train[train['outliers'] ==0]
target = train['target']
features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
train = train[features]
test = test[features]
#%%
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

corr_scores_df.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/feature_correlation_without_outliers.csv', index=False)

#%%
corr_scores_df = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/feature_correlation_without_outliers.csv')

threshold = 70
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


features = train.columns.values
categorical_feats = [c for c in features if 'feature_' in c]

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
sample_submission.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestline_submission_without_outliers.csv', index=False)

#%%
#-----------------------------------training for outlier likelihood-----------------------------
train = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/train_c.csv'))
test = reduce_mem_usage(pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/test_c.csv'))

target = train['outliers']
features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
train = train[features]
test = test[features]
#%%
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
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "nthread": 8,
         "is_unbalance":'true',
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

#%%
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

corr_scores_df.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/feature_correlation_outliers_likelihood.csv', index=False)

#%%

from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


corr_scores_df = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/feature_correlation_outliers_likelihood.csv')
threshold = 10
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


features = train.columns.values
categorical_feats = [c for c in features if 'feature_' in c]

param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'f1',
         "lambda_l1": 0.1,
         "nthread": 7,
         "verbosity": -1,
         "is_unbalance":'true'}

params ={
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'f1',
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
            'drop_seed':int(2**7),
            "is_unbalance":'true'
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

    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data],feval=lgb_f1_score ,verbose_eval=100, early_stopping_rounds = 400)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits


sample_submission = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/bestline_submission_outliers_likelihood2.csv', index=False)

#%%
model_without_outliers = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/bestline_submission_without_outliers.csv')
df_outlier_prob = pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/bestline_submission_outliers_likelihood.csv')
outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])
best_submission= pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/bestline_submission_main.csv')

most_likely_liers = best_submission.merge(outlier_id,how='right')

for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers["card_id"].isin(outlier_id["card_id"].values), "target"] = best_submission[best_submission["card_id"].isin(outlier_id["card_id"].values)]["target"]
#%%
model_without_outliers.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/combining_submission.csv', index=False)
