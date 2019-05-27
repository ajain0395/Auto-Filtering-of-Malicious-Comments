#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:10:28 2019

@author: piyush
"""

def convert(label):
    mod = []
    t = 0
    for i in range(len(label)):
        if label[i]==1:
            mod.append(i)
            t=1
    if t==0:
        mod.append(6)
    return mod
            
def convert1(label,prob):
    mod = []
    mod1=[]
    t = 0
    for i in range(len(label)):
        if label[i]==1:
            mod.append(i)
            mod1.append(prob[i])
            t=1
    if t==0:
        mod.append(6)
        mod1.append(prob[0])
    return (mod,mod1)



y_true = pickle.load(open("test_labels_nbbr.pkl", "rb"))
y_pred = pickle.load(open("pred_labels_nbbr.pkl", "rb"))
y_score = pickle.load(open("prob_score_nbbr.pkl", "rb"))
y_true = y_true.tolist()
y_pred = y_pred.tolist()
y_score = y_score.tolist()

truey=[]
for i in range(len(y_true)):
    truey.append(convert(y_true[i]))

scorey=[]
predy=[]
for i in range(len(y_pred)):
    A = convert1(y_pred[i],y_score[i])
    scorey.append(A[1])
    predy.append(A[0])
    
pickle.dump(truey,open("new_test_labels_nbbr.pkl","wb"))
pickle.dump(predy,open("new_pred_labels_nbbr.pkl","wb"))
pickle.dump(scorey,open("new_prob_score_nbbr.pkl","wb"))