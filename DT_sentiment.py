import csv
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils import shuffle

df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])

""" Functions for text pre-processing """


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", " ", sample)


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


try:
    size = int(sys.argv[1])
except IndexError:
    size = None

""" Data creation """
text_data = np.array([])
# Read tweets
for text in df.text:
    text_data = np.append(text_data, text)
# creating target classes
Y = np.array([])
for text in df.sentiment:
    Y = np.append(Y, text)

# First 1500 for training set, last 500 for test set
X_train, X_test, y_train, y_test = train_test_split(
    text_data, Y, test_size=0.25, shuffle=False)

count = CountVectorizer(preprocessor=myPreprocessor,
                        lowercase=False, tokenizer=myTokenizer, max_features=size)

X_train = count.fit_transform(X_train).toarray()
print("----------Train vector------------", len(X_train))
print(X_train)
X_test = count.transform(X_test).toarray()
print("----------Test vector------------", len(X_test))
print(X_test)

start_time = time.time()
# Decision Tree construction stops when a node covers 1 % (20) or fewer examples.
clf = tree.DecisionTreeClassifier(
    criterion='entropy', random_state=0, min_samples_leaf=20)
model = clf.fit(X_train, y_train)
training_time = (time.time() - start_time)

# print(y_test, y_pred)
# print(model.predict_proba(X_test))
# print(precision_score(y_test, y_pred, average='micro'))
# print(recall_score(y_test, y_pred, average='micro'))
# print(f1_score(y_test, y_pred, average='micro'))
# print(f1_score(y_test, y_pred, average='macro'))

y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
# print('Accuracy score:', accuracy_score(y_test, y_pred))
testtime = time.time() - start_time
test_report = classification_report(y_test, y_pred, output_dict=True)

start_time = time.time()
y_pred = model.predict(X_train)
# print(classification_report(y_train, y_pred))
# print('Accuracy score:', accuracy_score(y_train, y_pred))
trainingtime = (time.time() - start_time + training_time)


train_report = classification_report(y_train, y_pred, output_dict=True)

metric_list = ['precision', 'recall', 'f1-score']
avg_list = ['micro avg', 'macro avg', 'weighted avg']

test_str_output = "DT_sentiment\t"+f"{size}\t"+"test\t"
train_str_output = "DT_sentiment\t"+f"{size}\t"+"train\t"

for m in metric_list:
    for a in avg_list:
        test_str_output = test_str_output + f"{test_report[a][m]:.3f}\t"
        train_str_output = train_str_output + f"{train_report[a][m]:.3f}\t"
test_str_output += f"{testtime:.4f}"
train_str_output += f"{trainingtime:.4f}"
print(test_str_output.rstrip())
print(train_str_output.rstrip())
