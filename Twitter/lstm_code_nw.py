#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:55:44 2019

"""

import gc
import pickle 
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


def embedMeanStdMatrix(embed_index,embed_size):
    embs = np.stack(list(embed_index.values()))
    emb_mean,emb_std = embs.mean(), embs.std()
    nb_words = len(tokenizer.word_index)
    return np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    

def pickle_dump(data_to_dump,dump_name):
    pickle_open = open(dump_name,"wb")
    pickle.dump(data_to_dump, pickle_open)
    pickle_open.close()
    
def readingEmbeddedFile(file_path):
    indexing =dict()
    f = open(file_path)
    for line in f:
        vals = line.split()
        word = vals[0]
        coefficients = np.array(vals[1:], dtype='float32')
        indexing[word] = coefficients
    f.close()
    return indexing
    

def generatingEmbeddings(tokenizer):
    embed_size = 100
    print "loading pretrained embedded  file..."
    embed_file_path='Twitter_Dataset/Wikipedia-20190428T203336Z-001/Wikipedia/glove.twitter.27B.100d.txt'
    embed_index=readingEmbeddedFile(embed_file_path) 
    e_mat = embedMeanStdMatrix(embed_index,embed_size)
    for word, i in tokenizer.word_index.items():
        i-=1
        e_vec = embed_index.get(word)
        if e_vec is not None: 
                e_mat[i] = e_vec
    return e_mat

rows=[]
tweets=[]
label=[]
with open("./Twitter_Dataset/labeled_data.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    
    title=next(csvreader)
    for row in csvreader:
        rows.append(row)

for row in rows:
    tweets.append(row[6])
    label.append(int(row[5]))
    
mod_labels_lstm=[]
for lb in label:
    mod_labels_lstm.append((lb,))
    

mlb = MultiLabelBinarizer()
lstm_labels = mlb.fit_transform(mod_labels_lstm)   

#splitting of train and test data
train_comment,test_comment,trainlabel,testlabel = train_test_split(tweets,lstm_labels,test_size=0.10,random_state=42)

max_features = 10000
maxlen = 200
tokenizer = Tokenizer(num_words=max_features)
def tokenize_seq(max_features,train_comment,test_comment):
    tokenizer.fit_on_texts(list(train_comment))
    tokenized_train = tokenizer.texts_to_sequences(train_comment)
    tokenized_test = tokenizer.texts_to_sequences(test_comment)
    return tokenized_train,tokenized_test
#
def padSeq(maxlen,tokenized_train,tokenized_test):
    
    x_train = pad_sequences(tokenized_train, maxlen=maxlen)
    x_test = pad_sequences(tokenized_test, maxlen=maxlen)
    return x_train,x_test

tokenized_train,tokenized_test = tokenize_seq(max_features,train_comment,test_comment)
x_train,x_test = padSeq(maxlen,tokenized_train,tokenized_test)

y_train = trainlabel
y_test=testlabel
print "embedding ..."

# create the model
#embedding_vecor_length = 32
#model = Sequential()
#model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
#model.add(LSTM(100))
#model.add(Dense(3, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#model.fit(X_t, y, epochs=2, batch_size=64)
## Final evaluation of the model
#scores = model.evaluate(X_te, test_y, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

embedding_matrix = generatingEmbeddings(tokenizer)
 
#pickle_dump(embedding_matrix,"glove_embed.pickle")  

input_shape = Input(shape=(maxlen, ))

def Blstm(tokenizer):
    x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)
    
    x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(3, activation="sigmoid")(x)
    return x

out=Blstm(tokenizer)

model = Model(inputs=input_shape, outputs=out)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print model.summary()
#
#pickle_dump(model,"validation_model")

#############training 
batch_size = 32
epochs = 2
hist = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#
#pickle_dump(hist,"train_model")
test_lb=[]
for i in y_test:
    test_lb.append(np.argmax(i))
    
    
pred_prob = model.predict([x_test], batch_size=32, verbose=1)

pred_lb=[]
for i in pred_prob:
    pred_lb.append(np.argmax(i))

from sklearn.metrics import classification_report
X = classification_report(test_lb,pred_lb)
print X
#y_test_label = model.evaluate(X_te, test_y,batch_size=32)
#
#
#
#pickle_dump(y_test,"pred_class_prob")
