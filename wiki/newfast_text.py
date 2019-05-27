#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 03:38:43 2019

@author: piyush
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:



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


# In[3]:


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


#data_path="C:/Users/Dell-pc/Desktop/Thesis/Dataset_kaggle/"
#os.chdir(data_path)
df = pd.read_csv("train.csv", encoding = "ISO-8859-1")
print(df.head())


# In[ ]:





# In[7]:


df_toxic = df.drop(['id', 'comment_text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))


# In[8]:


X_train=df['comment_text']
y_train=df[categories]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:
##################################################################################



df1 = pd.read_csv("testtest.csv", encoding = "ISO-8859-1")
print(df1.head())


# In[ ]:





# In[7]:


df_toxic = df1.drop(['id', 'comment_text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df1['comment_text'] = df1['comment_text'].map(lambda com : clean_text(com))


# In[8]:


X_test=df1['comment_text']
y_test=df1[categories]

###################################################################################
# In[20]:


train_data1=list(X_train)
test_data1=list(X_test)


# In[21]:


conv="__label__"


# In[22]:


y_toxic=list(y_train['toxic'])
y_severe_toxic=list(y_train['severe_toxic'])
y_obscene=list(y_train['obscene'])
y_threat=list(y_train['threat'])
y_insult=list(y_train['insult'])
y_identity_hate=list(y_train['identity_hate'])



# In[29]:


y_toxic1=list(y_test['toxic'])
y_severe_toxic1=list(y_test['severe_toxic'])
y_obscene1=list(y_test['obscene'])
y_threat1=list(y_test['threat'])
y_insult1=list(y_test['insult'])
y_identity_hate1=list(y_test['identity_hate'])


# In[23]:


y_toxic[0]


# In[ ]:





# In[24]:


f = open("train_idenhate.txt", "w")
for i in range(len(train_data1)):
    s1=''
    #s1=s1+conv+str(y_toxic[i])+','+conv+str(y_severe_toxic[i])+','+conv+str(y_obscene[i])+','+conv+str(y_threat[i])+','+conv+str(y_insult[i])+','+conv+str(y_identity_hate[i])
    s1=s1+conv+str(y_identity_hate[i])
    s1=s1+' '+train_data1[i]+"\n"
    f.write(s1)

    
    


# In[30]:


f = open("test_idenhate.txt", "w")
for i in range(len(test_data1)):
    s1=''
    #s1=s1+conv+str(y_toxic1[i])+','+conv+str(y_severe_toxic1[i])+','+conv+str(y_obscene1[i])+','+conv+str(y_threat1[i])+','+conv+str(y_insult1[i])+','+conv+str(y_identity_hate1[i])
    s1=s1+conv+str(y_identity_hate1[i])
    s1=s1+' '+test_data1[i]+"\n"
    f.write(s1)



# In[ ]:
model = FastText()
model.supervised(input="train_tox.txt",output = 'abc1.txt',epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)
pred1 = model.predict_file("test_tox.txt")
prob1 = model.predict_proba_file("test_tox.txt",normalized = True)


# In[ ]:
model = FastText()
model.supervised(input="train_sevtox.txt",output = 'abc2.txt',epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)
pred2 = model.predict_file("test_sevtox.txt")
prob2 = model.predict_proba_file("test_sevtox.txt",normalized = True)

###############################################################

model = FastText()
model.supervised(input="train_obsc.txt",output = 'abc3.txt',epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)
pred3 = model.predict_file("test_obsc.txt")
prob3 = model.predict_proba_file("test_obsc.txt",normalized = True)

######################################################################

model = FastText()
model.supervised(input="train_threat.txt",output = 'abc4.txt',epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)
pred4 = model.predict_file("test_threat.txt")
prob4 = model.predict_proba_file("test_threat.txt",normalized = True)

##########################################################################

model = FastText()
model.supervised(input="train_insult.txt",output = 'abc5.txt',epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)
pred5 = model.predict_file("test_insult.txt")
prob5 = model.predict_proba_file("test_insult.txt",normalized = True)
######################################################################

model = FastText()
model.supervised(input="train_idenhate.txt",output = 'abc6.txt',epoch=25,lr=0.4,wordNgrams=5,verbose=2,minCount=1)
pred6 = model.predict_file("test_idenhate.txt")
prob6 = model.predict_proba_file("test_idenhate.txt",normalized = True)
###############################################################



for i in range(len(pred1)):
    pred6[i] = int(pred6[i][0])
    

for i in range(len(pred1)):
    pred1[i] = int(pred1[i][0])
    pred2[i] = int(pred2[i][0])
    pred3[i] = int(pred3[i][0])
    pred4[i] = int(pred4[i][0])
    pred5[i] = int(pred5[i][0])

for i in range(len(pred1)):
    prob6[i] = float(prob6[i][0][1])
for i in range(len(pred1)):
    prob1[i] = float(prob1[i][0][1]) 
    prob2[i] = float(prob2[i][0][1]) 
    prob3[i] = float(prob3[i][0][1]) 
    prob4[i] = float(prob4[i][0][1]) 
    prob5[i] = float(prob5[i][0][1]) 

pred = []
pred.append(pred1)
pred.append(pred2)
pred.append(pred3)
pred.append(pred4)
pred.append(pred5)
pred.append(pred6)

prob = []
prob.append(prob1)
prob.append(prob2)
prob.append(prob3)
prob.append(prob4)
prob.append(prob5)
prob.append(prob6)

true = []
true.append(y_toxic1)
true.append(y_severe_toxic1)
true.append(y_obscene1)
true.append(y_threat1)
true.append(y_insult1)
true.append(y_identity_hate1)

pred = np.array(pred).T
true = np.array(true).T
prob = np.array(prob).T

pickle.dump(true,open("test_labels_ft.pkl","wb"))
pickle.dump(pred,open("pred_labels_ft.pkl","wb"))
pickle.dump(prob,open("prob_score_ft.pkl","wb"))



