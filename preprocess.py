# 添加自定义词典
# content_id,content,subject,sentiment_value,sentiment_word
import pandas as pd
import jieba
train_data=pd.read_csv('data/train.csv')
# print(train_data['sentiment_word'].isnull().value_counts())
print(train_data['sentiment_word'].dropna().shape)