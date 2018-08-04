import pandas as pd 
import numpy as np 
import csv
import string

#PREPARE TEST_X AND TEST_Y
#UNCOMMENT THIS SECTION TO PREPARE DATA
#####################################################################################################################################
# with open('cwishareddataset/testset/english/News_Test.tsv','r') as file1:
# 	count=0
# 	for rows in file1:
# 		#print "Before , ",rows
# 		#print "After , ",rows.strip().split('\t')
# 		#print "After , ",rows.strip().split('\t')[1],rows.strip().split('\t')[4]
# 		# dataX=[rows.strip().split('\t')[1],rows.strip().split('\t')[4]]
# 		# dataY=rows.strip().split('\t')[-2]
# 		# print "After , ",rows.strip().split('\t')
# 		# print "\n Xparams , ",rows.strip().split('\t')[1],rows.strip().split('\t')[4]
# 		# print "\n Yparams , ",rows.strip().split('\t')[-2]
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

# 		file2=open('News_Test_proc.tsv','a')
# 		tsv_out=csv.writer(file2,delimiter='\t')
# 		tsv_out.writerow(Xparams)

# 		file3=open('News_Test_labels.tsv','a')
# 		tsv_out1=csv.writer(file3,delimiter='\t')
# 		tsv_out1.writerow(Yparams)
# 		count+=1
# print "Count , ",count

#SPLIT TEST_X INTO SENTENCE BEFORE, WORD AND SENTENCE AFTER WORD COLUMNS
#UNCOMMENT TO SPLIT THE DATA
#####################################################################################################################################
# with open('News_Test_proc.tsv','r') as file1:
# 	count=0
# 	for rows in file1:
		
		
# 		print "\n Rows , ",rows.strip().split('\t')
# 		sentence=rows.strip().split('\t')[0]
# 		word=rows.strip().split('\t')[1]
# 		cb=[rows.strip().split('\t')[0].split(rows.strip().split('\t')[1])[0]]
# 		ca=[rows.strip().split('\t')[0].split(rows.strip().split('\t')[1])[1]]
# 		word=[rows.strip().split('\t')[1]]
# 		print "\n Context before , ",cb
# 		print "\n Context after , ",ca

		
# 		if cb==['']:
# 			cb=['NOVAL']
			

# 		if ca==['']:
# 			ca=['NOVAL']		
	 	 	
	 	
# 	 	testData=[]
# 		for x in [cb,word,ca]:
# 			testData.append(''.join(x)) 

# 	 	file2=open('Context_Word_Context_Testing.tsv','a')
# 		tsv_out=csv.writer(file2,delimiter='\t')
# 		tsv_out.writerow(testData) 
# 		print testData

# 	 	count+=1
	 	
# print "Count , ",count





