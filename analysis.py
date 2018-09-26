# 数据统计
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei']
# content_id,content,subject,sentiment_value,sentiment_word
train_data=pd.read_csv('data/train.csv')
print(train_data['subject'].value_counts())
print(train_data['sentiment_value'].value_counts())
print(train_data.shape)

# sent_copy=train_data[train_data['sentiment_value'].isin([1,-1])]
# train_data=pd.concat([train_data,sent_copy],axis=0)
# train_data=pd.concat([train_data,sent_copy],axis=0)
# train_data=pd.concat([train_data,sent_copy],axis=0)
# print(train_data.shape)
# print(train_data['sentiment_value'].value_counts())
# print(train_data['subject'].value_counts())

train_data['sentiment_value'].value_counts().plot(kind='barh')
plt.show()


subject_labels=dict()
for i,x in enumerate(pd.unique(train_data['subject'])):
    subject_labels[x]=i
train_data['subject']=train_data['subject'].apply(lambda x:subject_labels[x])
# print(train_data['subject'])

# content_id
train_data=train_data.drop_duplicates(subset=['content_id'],keep='first')
print(train_data.shape)

print(train_data['sentiment_value'].value_counts())



test_data = pd.read_csv('data/test_public.csv')
print(test_data.shape)
test_data=test_data.drop_duplicates(subset=['content_id'],keep='first')
print(test_data.shape)


