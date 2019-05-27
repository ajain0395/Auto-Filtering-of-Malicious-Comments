#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:51:16 2019

@author: piyush
"""

import codecs
import json
import pickle

# In[2]:


import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#import seaborn as sns
from sklearn.model_selection import train_test_split
from pyfasttext import FastText
import os
import csv
from sklearn.metrics import accuracy_score
# In[3]:
from sklearn import metrics

def print_results(N,p,r):
    print("N\t",str(N))
    print("P@{}\t{:.3f}".format(1,p))
    print("R@{}\t{:.3f}".format(1,r))
    


# In[4]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


# In[5]:





# In[ ]:





# In[7]:
# In[8]:




# In[ ]:
##################################################################################
rows = []
label = []
tweets = []
with open("labeled_data.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    
    title=next(csvreader)
    for row in csvreader:
        rows.append(row)

for row in rows:
    tweets.append(row[6])
    label.append(int(row[5]))


for i in range(len(tweets)):
    tweets[i] = clean_text(tweets[i])
# In[ ]:
title=[title[2],title[3],title[4]]  
traindata,testdata,trainlabel,testlabel = train_test_split(tweets,
                                                           label,
                                                           test_size=0.20,random_state=42)



# In[7]:




# In[8]:





# In[21]:




conv="__label__"

# In[24]:


f = open("train.txt", "w")
for i in range(len(traindata)):
    s1=''
    #s1=s1+conv+str(y_toxic[i])+','+conv+str(y_severe_toxic[i])+','+conv+str(y_obscene[i])+','+conv+str(y_threat[i])+','+conv+str(y_insult[i])+','+conv+str(y_identity_hate[i])
    s1=s1+conv+str(trainlabel[i])
    s1=s1+' '+traindata[i]+"\n"
    f.write(s1)

    
    


# In[30]:


f = open("test.txt", "w")
for i in range(len(testdata)):
    s1=''
    #s1=s1+conv+str(y_toxic1[i])+','+conv+str(y_severe_toxic1[i])+','+conv+str(y_obscene1[i])+','+conv+str(y_threat1[i])+','+conv+str(y_insult1[i])+','+conv+str(y_identity_hate1[i])
    s1=s1+conv+str(testlabel[i])
    s1=s1+' '+testdata[i]+"\n"
    f.write(s1)


# In[25]:


# categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)
# X_train = train.comment_text
# X_test = test.comment_text


# In[26]:


#train_data=os.path.join(data_path,"train.txt")

#train_data="train.txt"

# In[32]:


#valid_data=os.path.join(data_path,"test.txt")

#valid_data="test.txt"
# In[41]:


#model=supervised(input=train_data,epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)


# In[42]:


#results=model.test(valid_data)


# In[43]:


#print("Results : ",results)


# In[ ]:
model = FastText()
model.supervised(input="train.txt",output = 'abc.txt',epoch=30,lr=0.1,wordNgrams=2,verbose=3,minCount=1)
pred1 = model.predict_file("test.txt")
prob1 = model.predict_proba_file("test.txt")

for i in range(len(pred1)):
    pred1[i] = int(pred1[i][0])
for i in range(len(pred1)):
    prob1[i] = float(prob1[i][0][1])


acc = accuracy_score(testlabel,pred1)
print("The Accuracy is")
print(acc)
print(metrics.classification_report(testlabel, pred1))
print("Confusion Matrix")
print(metrics.confusion_matrix(testlabel, pred1))
