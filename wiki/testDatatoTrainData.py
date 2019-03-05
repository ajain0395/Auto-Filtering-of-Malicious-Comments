#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#Ashish Jain(MT18052)
#Shubham Gupta(MT18055)
#Sarosh Hasan(MT18084)

import csv
import string

from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

pun = []
pun.append("\n")
pun.append("\r")
pun.append("\t")

labelfile = "test_labels.csv"
datafile = "test.csv"

# initializing the titles and rows list
fields = []
labels = []
data = []
finaldata = []

# reading csv file
with open(labelfile, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = csvreader.next()

    flag = True
    # extracting each data row one by one
    for row in csvreader:
        flag22 = True
        '''if(row[1] == '0' and row[6] == '0' and row[5] == '0' and row[4] == '0' and row[3] == '0' and row[2] == '0'):
            continue'''
        for k in range(1,len(row)):
                if(int(row[k]) < 0):
                    flag22 = False
                    break
        if(flag22):
            labels.append(row)
#    # get total number of rows 
    print("Total no. of rows: %d"%(len(labels)))
print "Mylabels ",labels
#outfile = open("out2","w")
#outfile.writelines(labels)

fname = ""

with open(datafile, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = csvreader.next()

    flag = True
    for row in csvreader:
        if(len(row) > 0):
            data.append(row)
    print("Total no. of rows: %d"%(len(data)))
    
print data[0]    
rows = []
c = 0
for lab in labels:
    print c
    c+=1
    for dat in data:
        if(dat[0] == lab[0]):
            #print "Yes"
            row = []
            row.append(dat[0])
            row.append(dat[1])
            for k in range(1,len(lab)):
                row.append(lab[k])
            rows.append(row)
            break

generatenonToxic = False
count = 0;
with open("testtest.csv", 'w') as writeFile:
    fieldnames = ["id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    writer = csv.DictWriter(writeFile, fieldnames=fieldnames)
    writer.writeheader()
    writer = csv.writer(writeFile)
    for row in rows:
        list = []
        writer.writerow(row)