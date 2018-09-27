# 添加自定义词典
# content_id,content,subject,sentiment_value,sentiment_word
import pandas as pd
import jieba
train_data=pd.read_csv('data/train.csv')
# print(train_data['sentiment_word'].isnull().value_counts())
sentiment_words=train_data['sentiment_word'].dropna()
with open('data/userdict.txt','w',encoding='utf-8') as out_data:
    for word in sentiment_words:
        out_data.write(word+'\n')