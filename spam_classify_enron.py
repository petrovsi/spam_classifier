#  download and unarchive any of the folders in the directory with the code
# – for example,  download and unarchive “enron1/” folder, 
# which contains 3672 legitimate (ham) emails and 1500 spam emails.

import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

stoplist = stopwords.words('english')



def init_lists(folder): # initialise_lists
    enc = 'utf-16'
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:

        f = open(folder + a_file, 'r', encoding='latin-1')
        a_list.append(f.read())
    f.close()
    return a_list

def preprocess(sentence): #words like goes and going to the same lemma go
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence)]

def get_features(text, setting):
    if setting=='bow': #bow=bag of words
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}
		
def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier
	






    
