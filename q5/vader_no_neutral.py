import csv
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn import metrics, tree
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils import shuffle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])
df = df[df.sentiment != 'neutral']

text_data = np.array([])
# Read tweets
for text in df.text:
    text_data = np.append(text_data, text)


def getSentiment(score):

    if(score['compound'] >= 0.05):
        return 'positive'
    elif(score['compound'] >= -0.05 and score['compound'] < 0.05):
        return 'neutral'
    else:
        return 'negative'


# creating target classes
Y = np.array([])
for text in df.sentiment:
    Y = np.append(Y, text)

score_data = np.array([])
analyser = SentimentIntensityAnalyzer()
start_time = time.time()
for sentence in text_data[:1072]:
    score = analyser.polarity_scores(sentence)
    score_data = np.append(score_data, getSentiment(score))


def count(array, text):
    count = 0
    for word in array:
        if word == text:
            count += 1
    return count


print(classification_report(Y[:1072], score_data))
print(accuracy_score(Y[:1072], score_data))
print("--- baseline %s seconds ---" %
      (time.time() - start_time))

print('neutral', count(score_data, 'neutral'))
print('positive', count(score_data, 'positive'))
print('negative', count(score_data, 'negative'))
