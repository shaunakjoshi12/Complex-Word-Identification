import pandas as pd 
import numpy as np 
import csv
import string

#PREPARE TRAIN_X AND TRAIN_Y
#UNCOMMENT THIS SECTION TO PREPARE DATA
#####################################################################################################################################
# with open('cwishareddataset/traindevset/english/News_Train.tsv','r') as file1:
# 	count=0
# 	for rows in file1:

# 		sentence=rows.strip().split('\t')[1]
# 		word=rows.strip().split('\t')[4]
# 		print "\n Sentence bfore split , ",sentence
# 		print "\n Word bfore split , ",word
		
# 		sentence=sentence.replace('\xe2\x80\x9c','')
# 		sentence=sentence.replace('\xe2\x80\x99','')
# 		sentence=sentence.replace('\xe2\x80\x9d','')
# 		word=word.replace('\xe2\x80\x9c','')
# 		word=word.replace('\xe2\x80\x99','')
# 		word=word.replace('\xe2\x80\x9d','')

# 		sentence=sentence.translate(string.maketrans('', ''),string.punctuation)
#  		word=word.translate(string.maketrans('', ''),string.punctuation)

#  		print "\n Sentence after split , ",sentence
# 		print "\n Word after split , ",word

# 		Xparams=[sentence,word]
# 		Yparams=rows.strip().split('\t')[-2]

# 		print "\n XParams : ",Xparams
# 		print "\n YParams : ",Yparams

# 		file2=open('News_Train_proc.tsv','a')
# 		tsv_out=csv.writer(file2,delimiter='\t')
# 		tsv_out.writerow(Xparams)

# 		file3=open('News_Train_labels.tsv','a')
# 		tsv_out1=csv.writer(file3,delimiter='\t')
# 		tsv_out1.writerow(Yparams)
# 		count+=1
# print "Count , ",count

#SPLIT TRAIN_X INTO SENTENCE BEFORE, WORD AND SENTENCE AFTER WORD COLUMNS
#UNCOMMENT TO SPLIT THE DATA
#######################################################################################################################################
# with open('News_Train_proc.tsv','r') as file1:
# 	count=0
# 	for rows in file1:
		
# 		
# 		print "\n Rows , ",rows.strip().split('\t')
# 		sentence=rows.strip().split('\t')[0]
# 		word=rows.strip().split('\t')[1]
# 		cb=[rows.strip().split('\t')[0].split(rows.strip().split('\t')[1])[0]]
# 		ca=[rows.strip().split('\t')[0].split(rows.strip().split('\t')[1])[1]]
# 		word=[rows.strip().split('\t')[1]]
# 		print "\n Context before , ",cb
# 		print "\n Context after , ",ca

# 		
# 		if cb==['']:
# 			cb=['NOVAL']
			

# 		if ca==['']:
# 			ca=['NOVAL']		
	 	 	
	 	
# 	 	trainData=[]
# 		for x in [cb,word,ca]:
# 			trainData.append(''.join(x)) 

# 	 	file2=open('Context_Word_Context.tsv','a')
# 		tsv_out=csv.writer(file2,delimiter='\t')
# 		tsv_out.writerow(trainData) 
# 		print trainData
# 	 	count+=1
	 	
# print "Count , ",count





