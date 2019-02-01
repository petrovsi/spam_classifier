import os, random
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
import spam_classify_enron as sf

def run_online(classifier, setting):
    while True:
        features = sf.get_features(input('Your new email: '), setting)
        if (len(features) == 0):
            break
        print (classifier.classify(features))

spam = sf.init_lists('enron1/spam/')
ham = sf.init_lists('enron1/ham/')
all_emails = [(email, 'spam') for email in spam]
all_emails += [(email, 'ham') for email in ham]
random.shuffle(all_emails)
print ('Corpus size = ' + str(len(all_emails)) + ' emails')
 
all_features = [(sf.get_features(email, ''), label) for (email, label) in all_emails]
train_set, test_set, classifier = sf.train(all_features, 1.0)

