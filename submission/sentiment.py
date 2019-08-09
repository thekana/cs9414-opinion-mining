import csv
import re
import sys
import warnings

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings('ignore')

df_data = pd.read_csv(sys.argv[1], sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                      header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])
df_test = pd.read_csv(sys.argv[2], sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                      header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])
""" Functions for text pre-processing """


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_punctuation(sample):
    """Remove punctuations from a sample string"""
    return re.sub(r'[^\w\s\&\#\@]', '', sample)


def myPreprocessor(sample):
    """Customized preprocessor"""
    sample = remove_URL(sample)
    sample = remove_punctuation(sample)
    return sample


def myTokenizer(sample):
    """Customized tokenizer"""
    new_words = []
    words = sample.split(' ')
    new_words = [word for word in words if len(word) >= 2 and not word.lower(
    ).startswith('au') and not word.lower().startswith('#aus')]
    return new_words


""" Train data """
train_data = np.array([])
# Read tweets
for text in df_data.text:
    train_data = np.append(train_data, text)
# creating target classes
Y = np.array([])
for text in df_data.sentiment:
    Y = np.append(Y, text)

""" Test data """
test_data = np.array([])
# Read tweets
for text in df_test.text:
    test_data = np.append(test_data, text)

count = CountVectorizer(preprocessor=myPreprocessor, tokenizer=myTokenizer,
                        max_features=1400, ngram_range=(1, 1), min_df=1, max_df=0.4)
X_train = count.fit_transform(train_data).toarray()
X_test = count.transform(test_data).toarray()

clf = MultinomialNB()
model = clf.fit(X_train, Y)


y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    print(df_test.instance[i], y_pred[i])
