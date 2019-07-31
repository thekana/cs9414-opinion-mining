import time
import pandas as pd
import csv
import sys
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree, metrics
from sklearn.utils import shuffle

df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])

# Perform shuffle
df = shuffle(df)
text_data = np.array([])
# Read tweets
for text in df.text:
    text_data = np.append(text_data, text)

""" Functions for text pre-processing """


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_punctuation(sample):
    """Remove punctuations from a sample string"""
    punctuations = r'''!"&'()*+,-./:;<=>?[\]^`{|}~'''
    no_punct = ""
    for char in sample:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def myPreprocessor(sample):
    """Customized preprocessor"""
    sample = remove_URL(sample)
    sample = remove_punctuation(sample)
    return sample


def myTokenizer(sample):
    """Customized tokenizer"""
    new_words = []
    words = sample.split(' ')
    new_words = [word for word in words if len(word) >= 2]
    return new_words


count = CountVectorizer(preprocessor=myPreprocessor,
                        lowercase=False, tokenizer=myTokenizer, max_features=1000)
bag_of_words = count.fit_transform(text_data)
# print(count.get_feature_names())
# print(count.vocabulary_)
X = bag_of_words.toarray()
# creating target classes
Y = np.array([])
for text in df.sentiment:
    Y = np.append(Y, text)
# First 1500 for training set, last 500 for test set
X_train = X[:1500]
X_test = X[1500:]
y_train = Y[:1500]
y_test = Y[1500:]

testSetX = X_test
testSetY = y_test
start_time = time.time()
clf = MultinomialNB()
model = clf.fit(X_train, y_train)
y_pred = model.predict(testSetX)
# print(y_test, y_pred)
# print(model.predict_proba(X_test))
# print(precision_score(y_test, y_pred, average='micro'))
# print(recall_score(y_test, y_pred, average='micro'))
# print(f1_score(y_test, y_pred, average='micro'))
# print(f1_score(y_test, y_pred, average='macro'))
print(classification_report(testSetY, y_pred))
print('Accuracy score:', accuracy_score(testSetY, y_pred))
print("--- %s seconds ---" % (time.time() - start_time))
