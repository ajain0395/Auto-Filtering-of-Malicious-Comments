#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:21:09 2019

@author: sarosh
"""

import pandas as pd
import json
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testtest.csv')

test_labels=test_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
test_comment = test_df.comment_text
train_comment = train_df.comment_text

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(train_comment))
list_tokenized_train = tokenizer.texts_to_sequences(train_comment)
list_tokenized_test = tokenizer.texts_to_sequences(test_comment)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


pickle_in = open("model_pickle","rb")
model=pickle.load(pickle_in)
pickle_in.close()

pickle_emm = open("glove_embed.pickle","rb")
embed=pickle.load(pickle_emm)

y_test = model.predict([X_te], batch_size=1024, verbose=1)

