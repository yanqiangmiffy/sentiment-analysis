# -*- coding: utf-8 -*-
# @Time    : 2018/9/27 22:02
# @Author  : quincyqiang
# @File    : 04_tfidf_lgb.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import generate_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

X,y_sub,y_sent,X_submit,labels_subject=generate_data()
X_train_sub,X_test_sub,y_train_sub,y_test_sub=train_test_split(X,y_sub,random_state=42)
X_train_sent,X_test_sent,y_train_sent,y_test_sent=train_test_split(X,y_sent,random_state=42)


def train():
    # 主题
    lgb_train_sub=lgb.Dataset(X_train_sub,y_train_sub)
    lgb_eval_sub=lgb.Dataset(X_test_sub,y_test_sub)

    params_sub= {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'multi_logloss'},
        # 'is_unbalance': True,
        # 'num_class': 10,
        # 'num_leaves': 31,
        # 'min_data_in_leaf': 1,
        # 'learning_rate': 0.1,
        # 'verbose':1
        'num_class': 10,
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'num_iteration': 100,
        'bagging_freq': 17,
        'verbose': 0
    }

    print('Start training...')
    gbm_sub = lgb.train(params_sub,
             lgb_train_sub,
             num_boost_round=50,
             valid_sets=lgb_eval_sub,
             early_stopping_rounds=10)
    return gbm_sub


def evaluate(gbm_model,X_test,y_test):
    y_pred=gbm_model.predict(X_test,num_iteration=gbm_model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)

    # 精确度=真阳性/（真阳性+假阳性）
    precision = precision_score(y_test, y_pred, pos_label=None, average='weighted')
    # 召回率=真阳性/（真阳性+假阴性）
    recall = recall_score(y_test, y_pred, pos_label=None, average='weighted')
    # F1
    f1 = f1_score(y_test, y_pred, pos_label=None, average='weighted')
    # 精确率
    accuracy = accuracy_score(y_test, y_pred)
    print("precision:{:.4f}-recall:{:.4f}-f1:{:.4f}-accuracy:{:.4f}".format(precision, recall, f1, accuracy))


def predict(gbm_model,X_submit,labels_subject):
    pred_submit=gbm_model.predict(X_submit,num_iteration=gbm_model.best_iteration)
    pred_submit = np.argmax(pred_submit, axis=1)
    pred_submit=[labels_subject[i] for i in list(pred_submit)]
    print(pred_submit)

if __name__ == '__main__':
    gbm_sub=train()
    # 测试集评价指标
    evaluate(gbm_sub, X_test_sub, y_test_sub)
    # 预测
    predict(gbm_sub,X_submit,labels_subject)