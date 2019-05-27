#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
import pandas as pd
import numpy as np
import copy
import seaborn as seab
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix, hstack

def add_feature(X, feature_to_add):
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testtest.csv')
#print train_df.describe()


# In[3]:


import warnings
warnings.filterwarnings('ignore')

print ('comments that are non toxic: '),
nontoxic=len(train_df[(train_df['toxic']==0) & (train_df['severe_toxic']==0) & (train_df['obscene']==0) & (train_df['threat']== 0) & (train_df['insult']==0) & (train_df['identity_hate']==0)])
print nontoxic 
print ('Percentage of comments that are non toxic:')
print (float(nontoxic)/len(train_df))*100

categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train_comment = train_df.comment_text
test_comment = test_df.comment_text
#print test_comment.shape,train_comment.shape

data = train_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
test_labels=test_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
#corr = data.astype(float).corr()
#seab.heatmap(corr,linewidths=0.4,linecolor='white',annot=True)

#vect = TfidfVectorizer(max_features=4000,stop_words='english')
#train_vec = vect.fit_transform(train_comment)
#test_vec = vect.transform(test_comment)
#Zlist = {}
#for clas in categories:
#    Zlist[clas] = np.array(test_df[clas].tolist())
#
#    
#
##print test_comment.shape,train_comment.shape
##print train_vec.shape
##Zlist = np.array(Zlist)
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#logreg = LogisticRegression(C=12.0)
#nbmodel = MultinomialNB()
##print Zlist
#
#
## In[4]:
#
#
#resultantMatrix = []
##resultantMatrix.append(['Labels','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
#resultantMatrix.append(['Classification Technique','Average Accuracy'])
#matindex = 0
#
########################################################################
#
#print "#logistic regression with binary relevance"
#
#
#matindex +=1
#resultantMatrix.append([])
#resultantMatrix[matindex].append("Logistic Regression Binary Relevance")
#for classes in categories:
#    print 'Processing',classes
#    y = train_df[classes]
#    Z = Zlist[classes]
#    logreg.fit(train_vec, y)
#    y_pred_X = logreg.predict(test_vec)
#    accuracy=accuracy_score(Z, y_pred_X)
#    print 'Training accuracy :',accuracy
#    resultantMatrix[matindex].append(accuracy)
#    mett=metrics.classification_report(Z,y_pred_X)
#    print mett
#    #print logreg.predict_proba(test_vec)
#    test_y_prob = logreg.predict_proba(test_vec)[:,1]
#    #print test_y_prob
#
#
## In[5]:
#
#
##dummy = []
########################################################################
#print "#Naive Bayes with binary relevance"
#matindex +=1
#resultantMatrix.append([])
#resultantMatrix[matindex].append("MultinomialNB Binary Relevance")
#for classes in categories:
#    print 'Processing',classes
#    y = train_df[classes]
#    Z = Zlist[classes]
#    #logreg.fit(train_vec, y)
#    model = MultinomialNB()
#    model.fit(train_vec, y)  
#    #y_pred_X = logreg.predict(test_vec)
#    y_pred_X= model.predict(test_vec)
#    #dummy.append(y_pred_X[0])
#    accuracy=accuracy_score(Z, y_pred_X)
#    print 'Training accuracy :',accuracy
#    resultantMatrix[matindex].append(accuracy)
#    mett=metrics.classification_report(Z,y_pred_X)
#    print mett
#    #test_y_prob = model.predict_proba(test_vec)[:,1]
#    #dummy.append(test_y_prob)
##print dummy
#
#
## In[6]:
#
#
########################################################################
#print "# Logistic regression with classifier chain"
#
#matindex +=1
#resultantMatrix.append([])
#resultantMatrix[matindex].append("Logistic Regression Classifier Chain")
#test_vec_chain = copy.deepcopy(test_vec)
#train_vec_chain = copy.deepcopy(train_vec)
#for classes in categories:
#    print 'Processing',classes
#    y = train_df[classes]
#    Z = Zlist[classes]
#    #model = MultinomialNB()
#    #logreg.fit(train_vec, y)
#    logreg.fit(train_vec_chain, y)  
#    y_pred_X = logreg.predict(test_vec_chain)
#    #y_pred_X= model.predict(test_vec_chain)
#    accuracy=accuracy_score(Z, y_pred_X)
#    print 'Training accuracy :',accuracy
#    resultantMatrix[matindex].append(accuracy)
#    mett=metrics.classification_report(Z,y_pred_X)
#    print mett    
#    #test_y=logreg.predict(test_vec)
#    #test_y_prob = model.predict_proba(test_vec)[:,1]
#    train_vec_chain = add_feature(train_vec_chain, y)
#    print('Shape of train_vec is now {}'.format(train_vec_chain.shape))
#    # chain current label predictions to test_train_vec
#    test_vec_chain = add_feature(test_vec_chain, y_pred_X)
#    print('Shape of test_train_vec is now {}'.format(test_vec_chain.shape))
#
#
## In[7]:
#
#
########################################################################
#print "#Naive Bayes with classifier chain"
#matindex +=1
#resultantMatrix.append([])
#resultantMatrix[matindex].append("MultinomialNB Classifier Chain")
#test_vec_chain = copy.deepcopy(test_vec)
#train_vec_chain = copy.deepcopy(train_vec)
#for classes in categories:
#    print 'Processing',classes
#    y = train_df[classes]
#    Z = Zlist[classes]
#    nbmodel.fit(train_vec_chain, y)  
#    y_pred_X= nbmodel.predict(test_vec_chain)
#    accuracy=accuracy_score(Z, y_pred_X)
#    print 'Training accuracy :',accuracy
#    resultantMatrix[matindex].append(accuracy)
#    mett=metrics.classification_report(Z,y_pred_X)
#    print mett    
#    train_vec_chain = add_feature(train_vec_chain, y)
#    print('Shape of train_vec is now {}'.format(train_vec_chain.shape))
#    # chain current label predictions to test_train_vec
#    test_vec_chain = add_feature(test_vec_chain, y_pred_X)
#    print('Shape of test_train_vec is now {}'.format(test_vec_chain.shape))
########################################################################    
#import csv
#with open("graph.csv", 'w') as writeFile:
#    writer = csv.DictWriter(writeFile, fieldnames=resultantMatrix[0])
#    writer = csv.writer(writeFile)
#    writer.writerow(resultantMatrix[0])
#    for row in range(1,len(resultantMatrix)):
#        roww = []
#        roww.append(resultantMatrix[row][0])
#        val = 0.0
#        for j in range(1,len(resultantMatrix[row])):
#            val += resultantMatrix[row][j]
#        roww.append(val / (len(resultantMatrix[row]) - 1))
#        writer.writerow(roww)


###########################keras glove,w2vec,fasttext################################################
import gc
import pickle 
def pickle_dump(data_to_dump,dump_name):
    pickle_open = open(dump_name,"wb")
    pickle.dump(data_to_dump, pickle_open)
    pickle_open.close()
    
#embed_size=0
def embeddingMatrix(typeToLoad):
    print "loading file..."
    EMBEDDING_FILE='/home/sarosh/Documents/IR/project/Wikipedia-20190428T203336Z-001/Wikipedia/glove.twitter.27B.100d.txt'
    embed_size = 100
    embeddings_index =dict()
    f = open(EMBEDDING_FILE)
    for line in f:
        #split up line into an indexed array
        values = line.split()
        #first index is word
        word = values[0]
        #store the rest of the values in the array as a new array
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs #50 dimensions
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    gc.collect()
    
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    nb_words = len(tokenizer.word_index)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    gc.collect()    
    embeddedCount = 0
    for word, i in tokenizer.word_index.items():
        i-=1
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')
    
    del(embeddings_index)
    gc.collect()
    return embedding_matrix

import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers



max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_comment))
list_tokenized_train = tokenizer.texts_to_sequences(train_comment)
list_tokenized_test = tokenizer.texts_to_sequences(test_comment)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
y = data.values
test_y=test_labels.values

embedding_matrix = embeddingMatrix('word2vec')
#### dumping embedding matrix   
#with open('glove_embed.json','w') as o:
#    json.dump(embedding_matrix,o) 
pickle_dump(embedding_matrix,"glove_embed.pickle")  
#pickle_out = open("glove_embed.pickle","wb")
#pickle.dump(embedding_matrix, pickle_out)
#pickle_out.close()
#    
inp = Input(shape=(maxlen, ))
print inp
x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
#x = Dense(6, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print model.summary()

pickle_dump(model,"validation_model")
#pickle_o = open("model_pickle","wb")
#pickle.dump(model, pickle_o)
#pickle_o.close()
############training 
batch_size = 32
epochs = 4
hist = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

pickle_dump(hist,"train_model")
#pickle_o = open("train_model","wb")
#pickle.dump(hist, pickle_o)
#pickle_o.close()

y_test = hist.predict([X_te], batch_size=1024, verbose=1)



pickle_dump(y_test,"pred_class_prob")