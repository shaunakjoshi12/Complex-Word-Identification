import pandas as pd 
import cPickle as pickle
import numpy as np 
from keras.layers import RepeatVector
from keras.layers import LSTM,Embedding,GRU,Dense,Input,Concatenate, TimeDistributed,Dropout
from keras.models import Sequential,Model
from keras.preprocessing import sequence
from keras.optimizers import Adam,RMSprop,SGD
from keras.callbacks import Callback,ModelCheckpoint
import datetime



#COUNTS THE VOCABULARY IN THE DATA
def vocab_size_counter(sentence_list):
	words_=[]

	for wrds in sentence_list:
		for ind_wrds in wrds.split():
			words_.append(ind_wrds)

	return [len(list(set(words_))),list(set(words_))]

#MAINTAINS THE WORD INDEX OF THE WORDS IN DATA
def word_indexer(unique_list):
	word_index={}
	index_word={}
	for index,word in enumerate(unique_list):
		word_index[word]=index
		index_word[index]=word

	return word_index

#PAD SEQUENCES TO SENTENCE LESSER IN LENGTH THAN MAX SENTENCE LENGTH
def sequence_padder(sentence_list,max_len,word_index):
	cnt=1
	new_sentence_list=[]
	for each_sentence in sentence_list:
		new_sentence_list.append([word_index[text] for text in each_sentence.split()])


	new_sentence_list=sequence.pad_sequences(new_sentence_list,maxlen=max_len,padding='post') 	
	
	return new_sentence_list

#CALCULATE MAX LENGTH OF SENTENCE
def max_len_counter(sentence_list):
	max_len=0

	

	for text in sentence_list:
		if len(text.split())> max_len:
			max_len=len(text.split())
		
		
	return max_len



#READ AND PREPARE TEST FEATURES
context_before_test=list()
word_test=list()
context_after_test=list()
with open('Preprocessed_Data/News_Test_Final_Features.tsv','r') as test_feat:
	cnt=0
	for rows in test_feat:
		testData=rows.strip().split('\t')
		context_before_test.append(testData[0])
		word_test.append(testData[1])
		context_after_test.append(testData[2])
		cnt+=1



unique_con_bfore=vocab_size_counter(context_before_test)


unique_con_after=vocab_size_counter(context_after_test)


unique_words=vocab_size_counter(word_test)


max_con_bfore=max_len_counter(context_before_test)
max_con_after=max_len_counter(context_after_test)
max_word=max_len_counter(word_test)

word_index_con_bfore=word_indexer(unique_con_bfore[1])
word_index_con_after=word_indexer(unique_con_after[1])
word_index_word=word_indexer(unique_words[1])

#READ MAX LENGTH OF SENTENCE FROM PICKLE FILE STORED IN TRAIN.PY
max_terms_dict=pickle.load(open("Max_terms.p","rb"))


context_before_test=sequence_padder(context_before_test,max_terms_dict['max_con_bfore'],word_index_con_bfore)
context_after_test=sequence_padder(context_after_test,max_terms_dict['max_con_after'],word_index_con_after)
word_test=sequence_padder(word_test,max_terms_dict['max_word'],word_index_word)

print "\n Shape of 'Context before the word ' test feature  ",context_before_test.shape
print "\n Shape of 'word' test feature ",word_test.shape
print "\n Shape of 'Context after the word ' test feature ",context_after_test.shape

#READ AND PREPARE TEST_Y NUMPY ARRAY
labels_test=[]
with open('Preprocessed_Data/News_Test_labels.tsv','r') as test_labels:
	cnt=0
	for rows in test_labels:
		labels_test.append(rows)

labels_test=np.asarray(labels_test)
labels_test=np.expand_dims(labels_test,axis=1)
print "\n Shape of test labels ",labels_test.shape


############################################################### MODEL ######################################################################################

#CONTEXT_BEFORE MODEL
##########################################################
l_context_input=Input(shape=(max_terms_dict['max_con_bfore'],))
embedding=Embedding(max_terms_dict['unique_con_bfore[0]'],128,input_length=max_terms_dict['max_con_bfore'])(l_context_input)
context_bfore_lstm_layer=LSTM(128,return_sequences=False)(embedding)
drop1=Dropout(0.5)(context_bfore_lstm_layer)
vector_context_l=RepeatVector(1)(drop1)
###########################################################



#WORDS  MODEL
###########################################################
word_input=Input(shape=(max_terms_dict['max_word'],))
embedding2=Embedding(max_terms_dict['unique_words[0]'],128,input_length=max_terms_dict['max_word'])(word_input)
words_lstm_layer=LSTM(128,return_sequences=False)(embedding2)
drop2=Dropout(0.5)(words_lstm_layer)
vector_word=RepeatVector(1)(drop2)
###########################################################


#CONTEXT AFTER MODEL
###########################################################
r_context_input=Input(shape=(max_terms_dict['max_con_after'],))
embedding3=Embedding(max_terms_dict['unique_con_after[0]'],128,input_length=max_terms_dict['max_con_after'])(r_context_input)
context_after_lstm_layer=LSTM(128,return_sequences=False)(embedding3)
drop3=Dropout(0.5)(context_after_lstm_layer)
vector_context_r=RepeatVector(1)(drop3)
###########################################################

#MERGED MODEL
###########################################################
merge_layer=Concatenate(axis=1)([vector_context_l,vector_word,vector_context_r])
LSTM_final=LSTM(1000,return_sequences=False,recurrent_dropout=0.2)(merge_layer)
drop4=Dropout(0.5)(LSTM_final)
###########################################################


pred_layer=Dense(1,activation='sigmoid')(drop4)

###########################################################

#CREATE AND COMPILE MODEL
###########################################################
model=Model([l_context_input,word_input,r_context_input],pred_layer)

print "\n Model summary "
print model.summary()

adamOpt=Adam(lr=0.02)
model.compile(optimizer=adamOpt,loss='binary_crossentropy',metrics=['accuracy'])
model.load_weights('CWIModelWeights/CWI_trained_model_weights.hdf5')

print "\n The accuracy on English News test dataset is: "
print model.evaluate(x=[context_before_test,word_test,context_after_test],y=labels_test)[1]*100

#SAVE TEST DATA FEATURES 
#########################################################################################################
# test_data_dict={}
# keys=['context_before_test','word_test','context_after_test','labels_test']
# list_save=[context_before_test,word_test,context_after_test,labels_test]
# cnt=0
# for key in keys:
# 	test_data_dict[key]=list_save[cnt]
# 	cnt+=1








