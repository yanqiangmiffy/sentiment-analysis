# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: main.py 
@Time: 2018/9/28 11:47
@Software: PyCharm 
@Description:
"""
from sentiment_classidication import main_sent
from subject_classification import main_sub
import pandas as pd


def submit():
    # 预测主题
    print("Training lgb for sub")
    # main_sub()
    # 预测情感
    print("Training lgb for sent")
    # main_sent()
    # 生成提交结果
    submit_data = pd.read_csv('../data/test_public.csv')
    df_sub=pd.read_csv('result/subject.csv')
    df_sent=pd.read_csv('result/sentiment.csv')

    submit_data['subject'] = df_sub['subject']
    submit_data['sentiment_value'] = df_sent['sentiment_value']
    submit_data['sentiment_word'] = None
    submit_data[['content_id', 'subject', 'sentiment_value', 'sentiment_word']].to_csv('result/lgb_submit.csv',
                                                                                       index=False)

if __name__ == '__main__':
    submit()