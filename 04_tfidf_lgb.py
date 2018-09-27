# -*- coding: utf-8 -*-
# @Time    : 2018/9/27 22:02
# @Author  : quincyqiang
# @File    : 04_tfidf_lgb.py
# @Software: PyCharm

import pandas as pd
import lightgbm as lgb
from utils import generate_data
from sklearn.model_selection import train_test_split
X,y_sub,y_sent,X_submit,labels_subject=generate_data()
X_train_sub,X_test_sub,y_train_sub,y_test_sub=train_test_split(X,y_sub,random_state=42)
X_train_sent,X_test_sent,y_train_sent,y_test_sent=train_test_split(X,y_sent,random_state=42)

def train():
    # 主题
    lgb_train_sub=lgb.Dataset(X_train_sub,y_train_sub)
    lgb_eval_sub=lgb.Dataset(X_test_sub,y_test_sub)

    params_sub= {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class':10,
        'metric':{'multi_logloss'},
              'nthread': -1,
              'silent': True,  # 是否打印信息，默认False
              'learning_rate': 0.1,
              'num_leaves': 80,
              'max_depth': 5,
              'max_bin': 127,
              'subsample_for_bin': 50000,
              'subsample': 0.8,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 1,
              'reg_lambda': 0,
              'min_split_gain': 0.0,
              'min_child_weight': 1,
              'min_child_samples': 20,
              'scale_pos_weight': 1}

    print('Start training...')
    gbm_sub = lgb.train(params_sub,
                        lgb_train_sub,
                    num_boost_round=20,
                    valid_sets=lgb_eval_sub,
                    early_stopping_rounds=5)

train()