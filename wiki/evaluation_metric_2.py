
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import coverage_error
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import zero_one_loss
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


def cal_int(label1,label2):
    sum=0
    for i in range(len(label1)):
        if label1[i]==1 and label2[i]==1:
            sum+=1
    return sum


# In[4]:


def cal_un(label1,label2):
    sum=0
    for i in range(len(label1)):
        if label1[i]==1 or label2[i]==1:
            sum+=1
    return sum


# In[5]:


def all_check(label):
    count=0
    for i in range(len(label)):
        if label[i]==0:
            count+=1
    if count==len(label):
        return True
    return False


# In[6]:


def auc_score(label2,score2):
    
    acc=[]
    for i in range(len(label2)):
        if all_check(label2[i]):
            acc.append(0)
            pass
        else:
            acc.append(roc_auc_score(np.array(label2[i]),np.array(score2[i]), average='macro', sample_weight=None))
    acc1=[]
    for i in range(len(label2)):
        if all_check(label2[i]):
            acc1.append(0)
            pass
        else:
            
            acc1.append(roc_auc_score(np.array(label2[i]),np.array(score2[i]), average='micro', sample_weight=None))
    return np.mean(acc),np.mean(acc1)


# # Accuracy

# In[7]:


###
def accuracy(label1,label2):
    sum=0
    for i in range(len(label1)):
        union=len(set(label1[i]).union(set(label2[i])))
        inter=len(set(label1[i]).intersection(set(label2[i])))
        try:
            sum+=inter/float(union)
        except:
            pass
    return sum/float(len(label1))


# In[8]:


def subset_acc_main(label1,label2,count):
    sum=0
    for i in range(len(label1)):
        
        tp=len(set(label1[i]).intersection(set(label2[i])))
        s1=symmetric_diff(label1[i],label2[i])
        tn=count-s1-tp
        try:
            sum+=(tp+tn)/float(count)
        except:
            pass
    return sum/float(len(label1))


# In[9]:


###
def count_no(label):
    count=0
    for i in range(len(label)):
        if label[i]==1:
            count+=1
    return count


# In[10]:


#label1 is true label and label2 is predict label
def precision_macro(label1,label2):
    sum=0
   
    inter=len(set(label1).intersection(set(label2)))
    union=len(label2)
    try:

        sum+=inter/float(union)
    except:
        pass
    return sum


# In[11]:


#label1 is true label and label2 is predict label
def precision(label1,label2):
    sum=0
    for i in range(len(label1)):
        inter=len(set(label1[i]).intersection(set(label2[i])))
        union=len(label2[i])
        try:
            
            sum+=inter/float(union)
        except:
            pass
    return sum/float(len(label1))


# In[12]:


#label1 is true label and label2 is predict label
def recall_macro(label1,label2):
    sum=0

    inter=len(set(label1).intersection(set(label2)))
    union=len(label1)
    try:
        sum+=inter/float(union)
    except:
        sum+=0
    return sum


# In[13]:


#label1 is true label and label2 is predict label
def recall(label1,label2):
    sum=0
    for i in range(len(label1)):
        inter=len(set(label1[i]).intersection(set(label2[i])))
        union=len(label1[i])
        try:
            sum+=inter/float(union)
        except:
            sum+=0
    return sum/float(len(label1))


# In[14]:


####
def f1measure(label1,label2):
    pre=precision(label1,label2)
    rec=recall(label1,label2)
 
    return (2*pre*rec)/float(pre+rec)


# In[15]:


####
def f1measure_macro(label1,label2):
    pre=precision_macro(label1,label2)
    rec=recall_macro(label1,label2)
 
    return (2*pre*rec)/float(pre+rec)


# In[16]:


####
def f_one_measure(label1,label2):
    sum=0
    for i in range(len(label1)):
        inter=cal_int(label1[i],label2[i])*2
        union1=count_no(label2[i])
        union2=count_no(label1[i])
        union=union1+union2
        try:
            
            sum+=inter/float(union)
        except:
            sum+=0
    return sum/float(len(label1))


# In[17]:


def symmetric_diff(label1,label2):
    
    d1=set(label2).symmetric_difference(set(label1))    
    return len(d1)


# In[18]:


# def auc_score(label,score):
#     label=np.transpose(np.array(label))
#     score=np.transpose(np.array(score))
#     auc_macro=[]
#     auc_micro=[]
#     for i in range(len(label)):
#         auc_macro.append(roc_auc_score(label[i],score[i], average='macro', sample_weight=None))
#         auc_micro.append(roc_auc_score(label[i],score[i], average='micro', sample_weight=None))
#     return np.mean(auc_macro),np.mean(auc_micro)


# In[19]:


def true_positive(true,pred):
    return len([(true[i]==pred[i] and true[i]==1) for i in range(len(pred))])


# In[20]:


def true_negative(true,pred):
    return len([(true[i]==pred[i] and true[i]==0) for i in range(len(pred))])


# In[21]:


def false_positive(true,pred):
    count=len([i==1 for i in range(len(pred))])
    return (count-true_positive(true,pred))


# In[22]:


def false_negative(true,pred):
    count=len([i==0 for i in range(len(pred))])
    return (count-true_negative(true,pred))


# In[23]:


# #################
# def label_macro_micro(label2,label1):
# #     label2=np.transpose(np.array(label2))
# #     label1=np.transpose(np.array(label1))
#     precision_macro1=0
#     recall_macro1=0
#     f1measure_macro1=0
#     tp1=0
#     tn1=0
#     fp1=0
#     fn1=0
#     for i in range(label_length):
#         print("hello",i)
#         true=[int(i in label2[j]) for j in range(len(label2))]
#         pred=[int(i in label1[j]) for j in range(len(label1))]
#         precision_macro1+=(precision_macro(true,pred))
#         recall_macro1+=(recall_macro(true,pred))
#         tp=true_positive(true,pred)
#         tn=true_negative(true,pred)
#         fp=false_positive(true,pred)
#         fn=false_negative(true,pred)
#         tp1+=(tp)
#         tn1+=(tn)
#         fp1+=(fp)
#         fn1+=(fn)
#         f1measure_macro1+=(f1measure_macro(true,pred))
        
#     precision_micro=np.sum(np.array(tp1))/float(np.sum(np.array(tp1))+np.sum(np.array(fp1)))
#     recall_micro=np.sum(np.array(tp1))/float(np.sum(np.array(tp1))+np.sum(np.array(fn1)))
#     f1_measure_micro=2*precision_micro*recall_micro/float(precision_micro+recall_micro)
#     return (precision_macro1/label_length),(recall_macro1/label_length),(f1measure_macro1/label_length),precision_micro,recall_micro,f1measure_micro


# In[24]:


def hamming_loss(label1,label2,label_length):
    q=label_length
    sum=0
    for i in range(len(label1)):    
        n=symmetric_diff(label1[i],label2[i])
        sum+=n/float(q)
    return sum/float(len(label1))


# In[25]:


def subset_0_1(label2,score):
    y_true=np.array(label2)
    y_pred=np.array(score)
    return zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None)


# In[26]:


label_length=7


# In[28]:


#pwd


# In[168]:

'''
direct1="./annexml-0.0.1/"
direct="./Toxic_Dataset/"
fpath1 = os.path.join(direct1, "label_predict_cal_toxic_annexml_aug01.txt")
fpath2 = os.path.join(direct1, "label_predict_score_cal_toxic_annexml_aug01.txt")
fpath3 = os.path.join(direct, "label_list_test_toxic.txt")
y_true = pickle.load(open(fpath3, "rb"))
y_pred = pickle.load(open(fpath1, "rb" ))
y_score= pickle.load(open(fpath2, "rb" ))
'''
print("LR-CC")
y_true = pickle.load(open("new_test_labels_ft.pkl", "rb"))
y_pred = pickle.load(open("new_pred_labels_ft.pkl", "rb"))
y_score = pickle.load(open("new_prob_score_ft.pkl", "rb"))
'''
y_true = y_true.tolist()
y_pred = y_pred.tolist()
y_score = y_score.tolist()
'''


# In[145]:


# direct="./WikipediaLarge-500K/"
# direct1="./Tree_Extreme_Classifiers/Parabel/"
# fpath1 = os.path.join(direct1, "label_predict_cal_wiki_parabel.txt")

# fpath2 = os.path.join(direct1, "label_predict_score_cal_wiki_parabel.txt")
# fpath3 = os.path.join(direct, "label_list_test_wiki.txt")
# y_true = pickle.load(open(fpath3, "rb"))
# y_pred = pickle.load(open(fpath1, "rb" ))
# y_score= pickle.load(open(fpath2, "rb" ))


# In[169]:


len(y_pred)


# In[170]:


len(y_true)


# In[171]:


len(y_score)


# In[172]:


#y_true=y_true[:151651]


# In[173]:


print("...................................Example Based Metrics ........................................")
print("1. Classification Metrics ")
print("Subset Accuracy : ",subset_acc_main(y_true,y_pred,label_length))


# In[174]:


print("Accuracy : ",accuracy(y_true,y_pred))


# In[175]:


print("Hamming Loss : ",hamming_loss(y_true,y_pred,label_length))


# In[176]:


print("Precision : ",precision(y_true,y_pred))


# In[177]:


print("Recall : ",recall(y_true,y_pred))


# In[178]:


print("F1 -measure : ",f1measure(y_true,y_pred))
print("2. Ranking Metrics ")


# In[179]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


# In[180]:


def coverage_cal(y_true,score,y_pred):
    sum=0
    for i in range(len(y_true)):
        r=y_true[i]
        s=[]
        c=1
        for j in range(len(r)):
            try:
                s.append(score[i].index(score[i][y_pred[i].index(r[j])]))
            except:
                p=0
                for k in range(len(y_pred[i])):
                    if y_pred[i][k]>r[j]:
                        p+=1
                
                s.append(r[j]+p)
#                 s.append(len(r)+c)
#                 c+=1
                pass
        try:
            sum+=np.max(np.array(s))
        except:
            pass
    return sum/float(len(y_true))


# In[181]:


#Done
print("Coverage  : ",coverage_cal(y_true,y_score,y_pred))


# In[182]:


def ranking_loss_cal(y_true,y_score,y_pred,label_length):
    sum=0
    for i in range(len(y_true)):
        count=0
        nr=list(set(y_pred[i]).difference(set(y_true[i])))
        r=list(set(y_pred[i]).intersection(set(y_true[i])))
        for j in range(len(nr)):
            for k in range(len(r)):
                if y_score[y_pred[i].index(nr[j])]>y_score[y_pred[i].index(r[k])]:
                    count+=1
        
        try:
            sum+=count/float(len(nr)*len(r))
        except:
            pass
    return sum/float(len(y_true))
        


# In[183]:


print("Ranking Loss : ",ranking_loss_cal(y_true,y_score,y_pred,label_length))


# In[184]:


def one_error(label2,score,y_pred,label_length):
    count=0
    for i in range(len(label2)):
        if y_pred[i][score[i].index(max(score[i]))] not in label2[i]:
            count+=1
    
    return count/float(len(label2))


# In[185]:


print("One error : ",one_error(y_true,y_score,y_pred,label_length))


# In[186]:


#Done
def average_precision(y_true,y_score,y_pred):
    prec=[]
    
    for i in range(len(y_true)):
        y_new=list(set(y_true[i]).intersection(set(y_pred[i])))
        sum=0
        y_sc=[]
        for j in range(len(y_new)):
            y_sc.append(y_score[i][y_pred[i].index(y_new[j])])
        
        for j in range(len(y_new)):
            pos=0
            pos=(np.sort(y_sc)[::-1].tolist()).index(y_sc[j])
            sum+=pos/(float((np.sort(y_score[i])[::-1].tolist()).index(y_score[i][y_pred[i].index(y_true[i][y_true[i].index(y_new[j])])]))+1)

        prec.append(sum/float(len(y_true[i])))
    
                
                
    return np.mean(prec)


# In[187]:


#Not done

print("Average Precision Scrore : ",average_precision(y_true,y_score,y_pred))
print("...................................Label Based Metrics ........................................")
print("1. Classification Metrics ")
# macro_precision,macro_recall,macro_f1measure,support=score(y_true1,y_pred1,average='macro')
# micro_precision,micro_recall,micro_f1measure,support=score(y_true1,y_pred1,average='micro')


# In[188]:


#################
def label_macro_micro(y_true,y_pred,label_length):
    result=[]
    for i in range(label_length):
        temp=[0]*3
        result.append(temp)
    for i in range(len(y_true)):
        inter=list(set(y_true[i]).intersection(set(y_pred[i])))
        if len(inter)!=0:

            for j in range(len(inter)):
                result[inter[j]][0]+=1
        fp=list(set(y_pred[i]).difference(set(y_true[i])))
        if len(fp)!=0:
        
            for j in range(len(fp)):
                result[fp[j]][1]+=1
        fn=list(set(y_true[i]).difference(set(y_pred[i])))
        if len(fn)!=0:
        
            for j in range(len(fn)):
                result[fn[j]][2]+=1
    #Macro
    macro_p=0
    macro_r=0
    macro_f=0
    for i in range(len(result)):
        
        macro_p+=result[i][0]/float(result[i][0]+result[i][1]+1)
        macro_r+=result[i][0]/float(result[i][0]+result[i][2]+1)
    macro_p=macro_p/float(label_length)
    macro_r=macro_r/float(label_length)
    macro_f=2*macro_p*macro_r/float(macro_p+macro_r)
      
    
    #Micro
    tp_val=0
    fp_val=0
    fn_val=0
    for i in range(len(result)):
        tp_val+=result[i][0]
        fp_val+=result[i][1]
        fn_val+=result[i][2]
    micro_p=tp_val/float(fp_val+tp_val+1)
    micro_r=tp_val/float(fn_val+tp_val+1)
    micro_f=2*micro_p*micro_r/(micro_p+micro_r)
    
        
    return macro_p,macro_r,macro_f,micro_p,micro_r,micro_f


# In[189]:


macro_precision,macro_recall,macro_f1measure,micro_precision,micro_recall,micro_f1measure=label_macro_micro(y_true,y_pred,label_length)
print("Micro Precision :",micro_precision)
print("Micro Recall :",micro_recall)
print("Micro F1 Measure :",micro_f1measure)
print("Macro Precision :",macro_precision)
print("Macro Recall :",macro_recall)
print("Macro F1 Measure :",macro_f1measure)


# In[165]:


def auc_score(y_true,y_pred,y_score,label_length):
    counter=0
    for i in range((label_length)):
        if counter%50==0:
            
            print("Counter : ",counter)
        if counter==2:
            print("Counter: ",counter)
        counter+=1
        sum1=0
        for j in range(len(y_true)):
            sum=0
            count=0
            if i in y_true[j] and i in y_pred[j]:
                for k in range(len(y_true)):
                    if i not in y_true[k] and i in y_pred[k]:
                        if y_score[j][y_pred[j].index(i)]>=y_score[k][y_pred[k].index(i)]:
                            sum+=1
                        count+=1
                    if i not in y_true[k] and i not in y_pred[k]:
                        sum+=1
                        count+=1
            if i in y_true[j] and i not in y_pred[j]:
                for k in range(len(y_true)):
                    if i not in y_true[k] and i not in y_pred[k]:
                        sum+=1
                    if i not in y_true[k]:
                        count+=1
                        
            sum1+=sum/float(count+1)
    auc_macro=sum1/float(label_length)
    return auc_macro
                
                        
    


# In[72]:


#################------------------>Not done
#print("...........AUC Score............. ")
#macro_auc=auc_score(y_true,y_pred,y_score,label_length)
#print("Macro AUC ",macro_auc)
# print("Micro AUC ",micro_auc)


# In[49]:


def precisionat3(y_true,y_pred,y_score,k):
    sum=0
    for i in range(len(y_true)):
        pred=[]
        pred=sort_list(y_pred[i],y_score[i])
        pred.reverse()
        sum1=0
        for j in range(k):
            if pred[j] in y_true[i]:
                sum1+=1
        sum+=sum1/float(k)
    return sum/len(y_true)


# In[50]:


def dcg_at_k(y_true,y_pred,y_score,k):
    sum=0
    for i in range(len(y_true)):
        pred=[]
        pred=sort_list(y_pred[i],y_score[i])
        pred.reverse()
        sum1=0
        for j in range(k):
            if pred[j] in y_true[i]:
                sum1+=1/np.log(j+2)
        sum+=sum1
    return sum/len(y_true)


# In[51]:


def ndcg_at_k(y_true,y_pred,y_score,k):
    s1=dcg_at_k(y_true,y_pred,y_score,k)
    s2=min(k,len(y_true))
    s=0
    for i in range(s2):
        s+=1/float(np.log(i+2))
    
    return s1/float(s)


# In[190]:


print("Precision @ 1 : ",precisionat3(y_true,y_pred,y_score,1))


# In[192]:


print("nDCGG @1 : ",ndcg_at_k(y_true,y_pred,y_score,1))

