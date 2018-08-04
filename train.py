import pandas as pd 
import numpy as np 
import cPickle as pickle
from keras.layers import LSTM,Embedding,GRU,Dense,Input,Concatenate, TimeDistributed,Dropout
from keras.models import Sequential,Model
from keras.preprocessing import sequence
from keras.layers import RepeatVector
from keras.callbacks import Callback,ModelCheckpoint
from keras.optimizers import Adam,RMSprop,SGD
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
	maxlist=[]
	minlist=[]
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


#READ AND PREPARE TRAIN FEATURES
context_before=list()
word=list()
context_after=list()
with open('Preprocessed_Data/News_Train_Final_Features.tsv','r') as train_feat:
	cnt=0
	for rows in train_feat:
		trainData=rows.strip().split('\t')
	
		context_before.append(trainData[0])
		word.append(trainData[1])
		context_after.append(trainData[2])
		
		cnt+=1



unique_con_bfore=vocab_size_counter(context_before)


unique_con_after=vocab_size_counter(context_after)


unique_words=vocab_size_counter(word)


max_con_bfore=max_len_counter(context_before)
max_con_after=max_len_counter(context_after)
max_word=max_len_counter(word)

word_index_con_bfore=word_indexer(unique_con_bfore[1])
word_index_con_after=word_indexer(unique_con_after[1])
word_index_word=word_indexer(unique_words[1])

list_to_be_saved=[]
dict_to_be_saved={}

keys=['max_con_bfore','max_con_after','max_word','unique_con_bfore[0]','unique_words[0]','unique_con_after[0]']
list_to_be_saved=[max_con_bfore,max_con_after,max_word,unique_con_bfore[0],unique_words[0],unique_con_after[0]]
counter=0
for key in keys:
	dict_to_be_saved[key]=list_to_be_saved[counter]
	counter+=1
print dict_to_be_saved

#DUMP MAX LENGTHS TO PICKLE FILE
with open("Max_terms.p","wb") as pickle_file:
	pickle.dump(dict_to_be_saved,pickle_file)

#READ TEST DATA DICTIONARY OF PREPARED TEST FEATURES FROM THE TEST FILE
test_data=pickle.load(open("Test_data.p","rb"))

context_before=sequence_padder(context_before,max_con_bfore,word_index_con_bfore)
context_after=sequence_padder(context_after,max_con_after,word_index_con_after)
word=sequence_padder(word,max_word,word_index_word)
print "\n Shape of 'Context before the word ' train feature  ",context_before.shape
print "\n Shape of 'word' train feature ",word.shape
print "\n Shape of 'Context after the word ' train feature ",context_after.shape

#READ AND PREPARE TRAIN_Y NUMPY ARRAY
labels=[]
with open('Preprocessed_Data/News_Train_labels.tsv','r') as train_labels:
	cnt=0
	for rows in train_labels:
		labels.append(rows)

labels=np.asarray(labels)
labels=np.expand_dims(labels,axis=1)
print "\n Shape of train labels ",labels.shape



############################################################### MODEL ######################################################################################

#CONTEXT_BEFORE MODEL
##########################################################
l_context_input=Input(shape=(max_con_bfore,))
embedding=Embedding(unique_con_bfore[0],128,input_length=max_con_bfore)(l_context_input)
context_bfore_lstm_layer=LSTM(128,return_sequences=False)(embedding)
drop1=Dropout(0.5)(context_bfore_lstm_layer)
vector_context_l=RepeatVector(1)(drop1)
###########################################################



#WORDS  MODEL
###########################################################
word_input=Input(shape=(max_word,))
embedding2=Embedding(unique_words[0],128,input_length=max_word)(word_input)
words_lstm_layer=LSTM(128,return_sequences=False)(embedding2)
drop2=Dropout(0.5)(words_lstm_layer)
vector_word=RepeatVector(1)(drop2)
###########################################################


#CONTEXT AFTER MODEL
###########################################################
r_context_input=Input(shape=(max_con_after,))
embedding3=Embedding(unique_con_after[0],128,input_length=max_con_after)(r_context_input)
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

print model.summary()
adamOpt=Adam(lr=0.02)
model.compile(optimizer=adamOpt,loss='binary_crossentropy',metrics=['accuracy'])
model.load_weights('CWIModelWeights/CWIWeights-dropout_0.5_dim128_intermediate_weights.hdf5')
###########################################################
############################################################################################################################################################


#TRAIN MODEL
###########################################################
file_to_save='CWIModelWeights/CWIWeights-dropout_0.5_dim128_{epoch:02d}.hdf5'
checkpoint=ModelCheckpoint(file_to_save,monitor='loss',verbose=1,save_best_only=False,mode='auto')
model.fit([context_before,word,context_after],labels,epochs=50,verbose=2,batch_size=32,validation_split=0.2,callbacks=[checkpoint])
###########################################################









