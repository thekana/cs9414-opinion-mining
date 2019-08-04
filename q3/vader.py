import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import tree, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import sys
import csv
import pandas as pd
from sklearn.dummy import DummyClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])


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
for sentence in text_data[1500:]:
    score = analyser.polarity_scores(sentence)
    score_data = np.append(score_data, getSentiment(score))
# print(score_data)
print(classification_report(Y[1500:], score_data))
print(accuracy_score(Y[1500:], score_data))
print("--- baseline %s seconds ---" %
      (time.time() - start_time))
