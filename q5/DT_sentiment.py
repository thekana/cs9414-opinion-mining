import csv
import re
import sys
import time

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import metrics, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils import shuffle

df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])
# Remove neutral tweets
df = df[df.sentiment != 'neutral']

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


def remove_stopwords_NLTK(sample):
    """Remove stopwords using NLTK"""
    stopWords = set(stopwords.words('english'))
    words = myTokenizer(sample)
    filteredText = ""
    for word in words:
        if word not in stopWords:
            filteredText = filteredText + word + " "
    return filteredText.rstrip()


def porter_stem(sample):
    """Stemming"""
    words = myTokenizer(sample)
    ps = PorterStemmer()
    stemmed_text = ""
    for word in words:
        stemmed_text = stemmed_text + ps.stem(word) + " "
    return stemmed_text.rstrip()


def myPreprocessor(sample):
    """Customized preprocessor"""
    sample = remove_URL(sample)
    sample = remove_stopwords_NLTK(sample)
    sample = remove_punctuation(sample)
    sample = porter_stem(sample)
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

count = CountVectorizer(preprocessor=myPreprocessor,
                        lowercase=False, tokenizer=myTokenizer, max_features=size)
bag_of_words = count.fit_transform(text_data)
# print(count.get_feature_names())
size = len(count.vocabulary_)
print(len(count.vocabulary_))
X = bag_of_words.toarray()
# creating target classes
Y = np.array([])
for text in df.sentiment:
    Y = np.append(Y, text)

# First 1072 for training set, leftover for test set

X_train = X[:1072]
X_test = X[1072:]
y_train = Y[:1072]
y_test = Y[1072:]

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
print(classification_report(y_test, y_pred))
# print('Accuracy score:', accuracy_score(y_test, y_pred))
testtime = time.time() - start_time
test_report = classification_report(y_test, y_pred, output_dict=True)

start_time = time.time()
y_pred = model.predict(X_train)
print(classification_report(y_train, y_pred))
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
