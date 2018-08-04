# Complex Word Identification task :  https://sites.google.com/view/cwisharedtask2018/call-for-participation
This project has solution for a specific part of Complex Word Identification Challenge. This model has been trained on English News Train data and tested on English News Test data tsv files as given in the dataset.
There are 14001 train examples in which 20% of the data has been separated for validation. The data has been tested on 2095 examples. 
In the dataset the parts of sentence before the word, the word itself and the sentence after the word each were passed to separate LSTM layers whose final outputs were merged and passed to another LSTM layer denser than the previous ones.

The tuned parameters of the model are: LSTM dimensions, dropout, learning rate of the adam optimizer, recurrent dropout of the final layer.
