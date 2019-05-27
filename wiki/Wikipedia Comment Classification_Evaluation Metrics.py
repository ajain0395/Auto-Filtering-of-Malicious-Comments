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
import pickle

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
print (nontoxic)
print ('Percentage of comments that are non toxic:')
print ((float(nontoxic)/len(train_df))*100)

categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
train_comment = train_df.comment_text
test_comment = test_df.comment_text
#print test_comment.shape,train_comment.shape

data = train_df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
#corr = data.astype(float).corr()
#seab.heatmap(corr,linewidths=0.4,linecolor='white',annot=True)

vect = TfidfVectorizer(max_features=4000,stop_words='english')
train_vec = vect.fit_transform(train_comment)
test_vec = vect.transform(test_comment)
Zlist = {}
for clas in categories:
    Zlist[clas] = np.array(test_df[clas].tolist())

    

#print test_comment.shape,train_comment.shape
#print train_vec.shape
#Zlist = np.array(Zlist)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)
nbmodel = MultinomialNB()
#print Zlist


# In[4]:


resultantMatrix = []
#resultantMatrix.append(['Labels','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
resultantMatrix.append(['Classification Technique','Average Accuracy'])
matindex = 0

#######################################################################

print ("#logistic regression with binary relevance")


matindex +=1
resultantMatrix.append([])
resultantMatrix[matindex].append("Logistic Regression Binary Relevance")
test_labels = []
pred_labels = []
prob_score = []
for classes in categories:
    print ('Processing',classes)
    y = train_df[classes]
    Z = Zlist[classes]
    logreg.fit(train_vec, y)
    y_pred_X = logreg.predict(test_vec)
    accuracy=accuracy_score(Z, y_pred_X)
    print ('Training accuracy :',accuracy)
    resultantMatrix[matindex].append(accuracy)
    mett=metrics.classification_report(Z,y_pred_X)
    print (mett)
    #print logreg.predict_proba(test_vec)
    test_y_prob = logreg.predict_proba(test_vec)[:,1]
    #print test_y_prob
    test_labels.append(Z)
    pred_labels.append(y_pred_X)
    prob_score.append(test_y_prob)
test_labels = np.array(test_labels).T
pred_labels = np.array(pred_labels).T
prob_score =  np.array(prob_score).T
pickle.dump(test_labels,open("test_labels_lrbr.pkl","wb"))
pickle.dump(pred_labels,open("pred_labels_lrbr.pkl","wb"))
pickle.dump(prob_score,open("prob_score_lrbr.pkl","wb"))

# In[5]:


#dummy = []
#######################################################################
print ("#Naive Bayes with binary relevance")
matindex +=1
resultantMatrix.append([])
resultantMatrix[matindex].append("MultinomialNB Binary Relevance")
test_labels = []
pred_labels = []
prob_score = []
for classes in categories:
    print ('Processing',classes)
    y = train_df[classes]
    Z = Zlist[classes]
    #logreg.fit(train_vec, y)
    model = MultinomialNB()
    model.fit(train_vec, y)  
    #y_pred_X = logreg.predict(test_vec)
    y_pred_X= model.predict(test_vec)
    #dummy.append(y_pred_X[0])
    accuracy=accuracy_score(Z, y_pred_X)
    print ('Training accuracy :',accuracy)
    resultantMatrix[matindex].append(accuracy)
    mett=metrics.classification_report(Z,y_pred_X)
    print (mett)
    test_y_prob = model.predict_proba(test_vec)[:,1]
    test_labels.append(Z)
    pred_labels.append(y_pred_X)
    prob_score.append(test_y_prob)
test_labels = np.array(test_labels).T
pred_labels = np.array(pred_labels).T
prob_score =  np.array(prob_score).T
pickle.dump(test_labels,open("test_labels_nbbr.pkl","wb"))
pickle.dump(pred_labels,open("pred_labels_nbbr.pkl","wb"))
pickle.dump(prob_score,open("prob_score_nbbr.pkl","wb"))
    #dummy.append(test_y_prob)
#print dummy


# In[6]:


#######################################################################
print ("# Logistic regression with classifier chain")

matindex +=1
resultantMatrix.append([])
resultantMatrix[matindex].append("Logistic Regression Classifier Chain")
test_vec_chain = copy.deepcopy(test_vec)
train_vec_chain = copy.deepcopy(train_vec)
test_labels = []
pred_labels = []
prob_score = []
for classes in categories:
    print ('Processing',classes)
    y = train_df[classes]
    Z = Zlist[classes]
    #model = MultinomialNB()
    #logreg.fit(train_vec, y)
    logreg.fit(train_vec_chain, y)  
    y_pred_X = logreg.predict(test_vec_chain)
    #y_pred_X= model.predict(test_vec_chain)
    accuracy=accuracy_score(Z, y_pred_X)
    print ('Training accuracy :',accuracy)
    resultantMatrix[matindex].append(accuracy)
    mett=metrics.classification_report(Z,y_pred_X)
    print (mett)   
    #test_y=logreg.predict(test_vec)
    test_y_prob = logreg.predict_proba(test_vec_chain)[:,1]
    train_vec_chain = add_feature(train_vec_chain, y)
    print('Shape of train_vec is now {}'.format(train_vec_chain.shape))
    # chain current label predictions to test_train_vec
    test_vec_chain = add_feature(test_vec_chain, y_pred_X)
    print('Shape of test_train_vec is now {}'.format(test_vec_chain.shape))
    test_labels.append(Z)
    pred_labels.append(y_pred_X)
    prob_score.append(test_y_prob)
test_labels = np.array(test_labels).T
pred_labels = np.array(pred_labels).T
prob_score =  np.array(prob_score).T
pickle.dump(test_labels,open("test_labels_lrcc.pkl","wb"))
pickle.dump(pred_labels,open("pred_labels_lrcc.pkl","wb"))
pickle.dump(prob_score,open("prob_score_lrcc.pkl","wb"))


# In[7]:


#######################################################################
print ("#Naive Bayes with classifier chain")
matindex +=1
resultantMatrix.append([])
resultantMatrix[matindex].append("MultinomialNB Classifier Chain")
test_vec_chain = copy.deepcopy(test_vec)
train_vec_chain = copy.deepcopy(train_vec)
test_labels = []
pred_labels = []
prob_score = []
for classes in categories:
    print ('Processing',classes)
    y = train_df[classes]
    Z = Zlist[classes]
    nbmodel.fit(train_vec_chain, y)  
    y_pred_X= nbmodel.predict(test_vec_chain)
    accuracy=accuracy_score(Z, y_pred_X)
    print ('Training accuracy :',accuracy)
    resultantMatrix[matindex].append(accuracy)
    mett=metrics.classification_report(Z,y_pred_X)
    print (mett)
    test_y_prob = nbmodel.predict_proba(test_vec_chain)[:,1]
    train_vec_chain = add_feature(train_vec_chain, y)
    print('Shape of train_vec is now {}'.format(train_vec_chain.shape))
    # chain current label predictions to test_train_vec
    test_vec_chain = add_feature(test_vec_chain, y_pred_X)
    print('Shape of test_train_vec is now {}'.format(test_vec_chain.shape))
    test_labels.append(Z)
    pred_labels.append(y_pred_X)
    prob_score.append(test_y_prob)
test_labels = np.array(test_labels).T
pred_labels = np.array(pred_labels).T
prob_score =  np.array(prob_score).T
pickle.dump(test_labels,open("test_labels_nbcc.pkl","wb"))
pickle.dump(pred_labels,open("pred_labels_nbcc.pkl","wb"))
pickle.dump(prob_score,open("prob_score_nbcc.pkl","wb"))
#######################################################################    
import csv
with open("graph.csv", 'w') as writeFile:
    writer = csv.DictWriter(writeFile, fieldnames=resultantMatrix[0])
    writer = csv.writer(writeFile)
    writer.writerow(resultantMatrix[0])
    for row in range(1,len(resultantMatrix)):
        roww = []
        roww.append(resultantMatrix[row][0])
        val = 0.0
        for j in range(1,len(resultantMatrix[row])):
            val += resultantMatrix[row][j]
        roww.append(val / (len(resultantMatrix[row]) - 1))
        writer.writerow(roww)