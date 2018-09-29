# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:yanqiang
@File: utils.py
@Time: 2018/9/26 14:31
@Software: PyCharm
@Description: 加载数据
"""
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = open('D:\Project\sentiment-analysis\data\stop_words.txt', 'r', encoding='utf-8').read().split('\n')


def word_seg(content):
    sent_words=[]
    for word in jieba.cut(content):
        if word not in stop_words and word!=' ' and len(word)>1:
            sent_words.append(word)
    return sent_words


def load_data(use_sina=False):
    # 内容分词
    train_data=pd.read_csv('D:\Project\sentiment-analysis\data\\train.csv')
    # 去除重复的ID
    train_data=train_data.drop_duplicates(subset=['content_id'],keep='first') # 去除重复id的数据
    train_data['word_seg']=train_data['content'].apply(lambda x:" ".join(word_seg(x)))
    print("train_data shpe:",train_data.shape)
    # 是否使用新浪数据
    if use_sina:
        df_sina = pd.read_csv('../spider/sina_auto.csv')
        df_sina['word_seg'] = df_sina['content'].apply(lambda x:" ".join(word_seg(x)))
        train_data=train_data.append(df_sina)
        print("train_data shape after using sina:", train_data.shape)
    submit_data=pd.read_csv('D:\Project\sentiment-analysis\data\\test_public.csv')
    submit_data['word_seg']=submit_data['content'].apply(lambda x:" ".join(word_seg(x)))
    # 提取tfidf特征
    return train_data,submit_data


def generate_data(use_sina=False):
    train_data, submit_data = load_data(use_sina)
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.9, use_idf=True, smooth_idf=True, sublinear_tf=True)
    X = vec.fit_transform(train_data['word_seg'])
    X_submit = vec.transform(submit_data['word_seg'])

    subject_labels = dict()
    labels_subject = dict()
    for i, x in enumerate(pd.unique(train_data['subject'])):
        subject_labels[x] = i
        labels_subject[i] = x
    train_data['subject'] = train_data['subject'].apply(lambda x: subject_labels[x])
    y_sub = train_data['subject'].astype(int)
    y_sent = train_data['sentiment_value'].astype(int)

    return X,y_sub,y_sent,X_submit,labels_subject