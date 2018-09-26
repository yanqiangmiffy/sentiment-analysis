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

stop_words = open('data/stop_words.txt', 'r', encoding='utf-8').read().split('\n')


def word_seg(content):
    sent_words=[]
    for word in jieba.cut(content):
        if word not in stop_words and word!=' ' and len(word)>1:
            sent_words.append(word)
    return sent_words


def load_data():
    # 内容分词
    train_data=pd.read_csv('data/train.csv')
    # 去除重复的ID
    train_data=train_data.drop_duplicates(subset=['content_id'],keep='first') # 去除重复id的数据
    train_data['word_seg']=train_data['content'].apply(lambda x:" ".join(word_seg(x)))
    submit_data=pd.read_csv('data/test_public.csv')
    submit_data['word_seg']=submit_data['content'].apply(lambda x:" ".join(word_seg(x)))
    # 提取tfidf特征
    return train_data,submit_data