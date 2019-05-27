# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 03:43:18 2019

@author: Legen
"""

import csv
import nltk
from scipy.sparse import coo_matrix, hstack
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import textstat.textstat 
#sentiment_analyzer = VS()


title=[]
rows=[]
tweets=[]
label=[]


def preprocessing(text):
         res=""
         val = text.lower()
         tokenizer=RegexpTokenizer(r'\w+')
         var=tokenizer.tokenize(val)
         stop_Words = set(stopwords.words('english'))
         newWords=[]
         for v in var:
             if v not in stop_Words:
                 newWords.append(v)
         p=PorterStemmer()
         for i in range(len(newWords)):
            newWords[i]=p.stem(newWords[i])
         for i in newWords:
             res=res+i+" "
             
         return res


def otherfeatures(tweet):
    #sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = preprocessing(tweet)
    syllables = textstat.syllable_count(words) #count syllables in words
    num_chars = sum(len(w) for w in words) #num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    FKRA = round(float(0.39 * float(num_words)/1.0))
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    features = [FKRA, FRE]
    return features



def getotherfeatures(tweets):
    total=[]
    for t in tweets:
        total.append(otherfeatures(t))
    return np.array(total)



with open("labeled_data.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    
    title=next(csvreader)
    for row in csvreader:
        rows.append(row)

for row in rows:
    tweets.append(row[6])
    label.append(int(row[5]))

title=[title[2],title[3],title[4]]    
'''
traindata=[]
trainlabel=[]
testdata=[]
testlabel=[]
upto=len(rows)*.75


# Partioning of data
for i in range(upto): #for training Data
    traindata.append(tweets[i])
    trainlabel.append(label[i])

for i in range(upto,len(rows)): #for test data
    testdata.append(tweets[i])
    testlabel.append(label[i])

'''     
#splitting of train and test data
traindata,testdata,trainlabel,testlabel = train_test_split(tweets,
                                                           label,
                                                           test_size=0.30,random_state=42)
train=[]                                                           
for i in traindata:
    temp = preprocessing(i)
    train.append(temp)

tf = TfidfVectorizer(ngram_range=(1,3), min_df = 0)
tf_idftrain = tf.fit_transform((train))
features = tf.get_feature_names()
newfeature = getotherfeatures(traindata)
newfeature = coo_matrix(newfeature)
tf_idftrain = hstack((tf_idftrain,newfeature))

'''
for i in range((tf_idftrain.shape[0])):
    tf_idftrain[i] =  np.matrix(list(tf_idftrain[i])+list(newfeature[i]))
'''


test=[]                                                           
for i in testdata:
    temp = preprocessing(i)
    test.append(temp)

tf_idftest = tf.transform((test))
newfeature = getotherfeatures(testdata)
newfeature = coo_matrix(newfeature)
tf_idftest = hstack((tf_idftest,newfeature))
'''
for i in range((tf_idftest.shape[0])):
    tf_idftest[i] = np.array(list(tf_idftest[i])+list(newfeature[i]))
'''    

''' 
# Using Naive Bayes as Claasifier
print("Using Naive Bayes as Classifier")
clf = MultinomialNB().fit(tf_idftrain,trainlabel)

pred = clf.predict(tf_idftest)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)
print(metrics.classification_report(testlabel, pred,
         target_names=title))
print("Confusion Matrix")
print(metrics.confusion_matrix(testlabel, pred))

# Using Linear SVM as Classifier
print("Using Linear SVM as Classifier")
sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, 
                    random_state=42,max_iter=5, tol=None).fit(tf_idftrain,trainlabel)

pred = sgd.predict(tf_idftest)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)
print(metrics.classification_report(testlabel, pred,
         target_names=title))
print("Confusion Matrix")
print(metrics.confusion_matrix(testlabel, pred))
'''

# Using Logistic Regression as Classifier
print("Using Logistic Regression as Classifier")
lg = LogisticRegressionCV().fit(tf_idftrain,trainlabel)
pred = lg.predict(tf_idftest)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)
print(metrics.classification_report(testlabel, pred,
         target_names=title))
print("Confusion Matrix")
print(metrics.confusion_matrix(testlabel, pred))







'''

fg = svm.SVC(gamma=0.001, C=100.).fit(tf_idftrain,trainlabel)
pred = fg.predict(tf_idftest)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)
print(metrics.classification_report(testlabel, pred,
         target_names=title))
print("Confusion Matrix")
print(metrics.confusion_matrix(testlabel, pred))



cl = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1).fit(tf_idftrain,trainlabel)
pred = cl.predict(tf_idftest)
acc = accuracy_score(testlabel,pred)
print("The Accuracy is")
print(acc)
print(metrics.classification_report(testlabel, pred,
         target_names=title))
print("Confusion Matrix")
print(metrics.confusion_matrix(testlabel, pred))
'''