from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score
from sklearn.svm import LinearSVC
import numpy as np
from utils import load_data
import pandas as pd
from gensim.models.word2vec import Word2Vec

data,submit_data=load_data()
sent_copy=data[data['sentiment_value'].isin([1,-1])]
# data=pd.concat([data,sent_copy],axis=0)
# data=pd.concat([data,sent_copy],axis=0)
# data=pd.concat([data,sent_copy],axis=0)

# corpus_len=[len(sent) for sent in corpus]
# print(max(corpus_len))
# print(data['word_seg'].shape)
# print(submit_data['word_seg'].shape)
# print(corpus.shape)


def build_sentence_vector(text,size,w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def load_data(n_dim=50):
    corpus = pd.concat([data['word_seg'], submit_data['word_seg']], axis=0)
    corpus = [sent.split(' ') for sent in corpus.tolist()]
    w2v=Word2Vec(corpus,size=n_dim,min_count=1)
    data_sentences=[sent.split(' ') for sent in data['word_seg']]
    X=np.concatenate([build_sentence_vector(text,n_dim,w2v) for text in data_sentences])
    y = data['sentiment_value'].astype(int)
    print(X.shape,y.shape)
    return X,y


X,y=load_data(n_dim=300)


def train_model(X,y):
    print("Training Model..")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
    clf=LinearSVC()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    report=classification_report(y_test,y_pred)
    print(report)

    # 精确度=真阳性/（真阳性+假阳性）
    precision = precision_score(y_test, y_pred, pos_label=None, average='weighted')
    # 召回率=真阳性/（真阳性+假阴性）
    recall = recall_score(y_test, y_pred, pos_label=None, average='weighted')
    # F1
    f1 = f1_score(y_test, y_pred, pos_label=None, average='weighted')
    # 精确率
    accuracy = accuracy_score(y_test, y_pred)
    print("precision:{:.4f}-recall:{:.4f}-f1:{:.4f}-accuracy:{:.4f}".format(precision, recall, f1, accuracy))

train_model(X,y)