from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
import jieba
import pandas as pd
import numpy as np
stop_words = open('data/stop_words.txt','r',encoding='utf-8').read().split('\n')


def word_seg(content):
    return [word for word in jieba.cut(content) if word not in stop_words]


# 内容分词
train_data=pd.read_csv('data/train.csv')
train_data['word_seg']=train_data['content'].apply(lambda x:" ".join(word_seg(x)))
test_data=pd.read_csv('data/test_public.csv')
test_data['word_seg']=test_data['content'].apply(lambda x:" ".join(word_seg(x)))

# 提取tfidf特征
vec = TfidfVectorizer(ngram_range=(1,3),min_df=1, max_df=0.9,use_idf=True,smooth_idf=True, sublinear_tf=True)
X_train_feature=vec.fit_transform(train_data['word_seg'])
X_test_feature=vec.transform(test_data['word_seg'])

# --------------情感值预测开始------------------------
y_train_sent=train_data['sentiment_value'].astype(int)
X_train_sent,X_test_sent,y_train_sent,y_test_sent=\
    train_test_split(X_train_feature,y_train_sent,test_size=0.1,random_state=42)
# clf = LogisticRegression(C=4, dual=True)
# clf =svm.LinearSVC()
# clf =RandomForestClassifier()
clf =SGDClassifier()

clf.fit(X_train_sent, y_train_sent)


# 在训练集评估模型
pred_test_sent=clf.predict(X_test_sent)
# 精确度=真阳性/（真阳性+假阳性）
precision=precision_score(y_test_sent,pred_test_sent,pos_label=None,average='weighted')
# 召回率=真阳性/（真阳性+假阴性）
recall=recall_score(y_test_sent,pred_test_sent,pos_label=None,average='weighted')
# F1
f1=f1_score(y_test_sent,pred_test_sent,pos_label=None,average='weighted')
# 精确率
accuracy=accuracy_score(y_test_sent,pred_test_sent)
print("precision:{:.4f}-recall:{:.4f}-f1:{:.4f}-accuracy:{:.4f}".format(precision,recall,f1,accuracy))


# 加载测试集以及预测
preds=clf.predict(X_test_feature)
# 两种效果写法，和上面一样
# preds=clf.predict_proba(X_test)
# preds=np.argmax(preds,axis=1)
# print([clf.classes_[pred] for pred in preds])
# --------------情感值预测结束------------------------



# -----------------主题预测开始----------------------
subject_labels=dict()
labels_subject=dict()
for i,x in enumerate(pd.unique(train_data['subject'])):
    subject_labels[x]=i
    labels_subject[i]=x
train_data['subject']=train_data['subject'].apply(lambda x:subject_labels[x])

y_train_sub=train_data['subject'].astype(int)
X_train_sub,X_test_sub,y_train_sub,y_test_sub=\
    train_test_split(X_train_feature,y_train_sub,test_size=0.1,random_state=42)
# sub_clf = LogisticRegression(C=10.0, solver='newton-cg', multi_class='multinomial')
# clf =svm.LinearSVC()
clf =SGDClassifier()
clf.fit(X_train_sub, y_train_sub)

# 在训练集评估模型
pred_test_sub=clf.predict(X_test_sub)
# 精确度=真阳性/（真阳性+假阳性）
precision=precision_score(y_test_sub,pred_test_sub,pos_label=None,average='weighted')
# 召回率=真阳性/（真阳性+假阴性）
recall=recall_score(y_test_sub,pred_test_sub,pos_label=None,average='weighted')
# F1
f1=f1_score(y_test_sub,pred_test_sub,pos_label=None,average='weighted')
# 精确率
accuracy=accuracy_score(y_test_sub,pred_test_sub)
print("precision:{:.4f}-recall:{:.4f}-f1:{:.4f}-accuracy:{:.4f}".format(precision,recall,f1,accuracy))


# 加载测试集以及预测
sub_preds=clf.predict(X_test_feature)
sub_preds=[labels_subject[i] for i in sub_preds]
# -----------------主题预测结束----------------------


# -------------------提交结果开始-----------------------
test_data['subject']=sub_preds
test_data['sentiment_value']=preds
test_data['sentiment_word']=None
test_data[['content_id','subject','sentiment_value','sentiment_word']].to_csv('result/01_tfidf_lr.csv',index=False)
# -------------------提交结果结束-----------------------
