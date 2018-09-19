# 数据统计

import pandas as pd
import matplotlib.pyplot as plt
# content_id,content,subject,sentiment_value,sentiment_word
train_data=pd.read_csv('data/train.csv')
# train_data['subject'].value_counts().plot(kind='barh')
# plt.show()
print(train_data['subject'].value_counts())
print(train_data['sentiment_value'].value_counts())


subject_labels=dict()
for i,x in enumerate(pd.unique(train_data['subject'])):
    subject_labels[x]=i
train_data['subject']=train_data['subject'].apply(lambda x:subject_labels[x])
print(train_data['subject'])