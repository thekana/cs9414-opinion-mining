import re
import sys
import nltk
import numpy as np
import pandas as pd
import csv
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)


def remove_punctuation(sample):
    """Remove punctuations from a sample string"""
    punctuations = '''!"&'()*+,-./:;<=>?[\]^`{|}~'''
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


df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE , dtype= str)

text = np.array(['This is a sentence ending here. This is another short, sweet sentence!This is a sentence with money $1,000,000. This is a sentence with hashtag #hashtag',
                 '#auspol @TurnbullMalcolm @Barnaby_Joyce We need an investigation https://t.co/fhc5t68jmM'])
count = CountVectorizer(preprocessor=myPreprocessor, tokenizer=myTokenizer)
bag_of_words = count.fit_transform(text)
print(count.get_feature_names())
print(count.vocabulary_)
X = bag_of_words.toarray()
print(X)
# from nltk import FreqDist, NaiveBayesClassifier
# from nltk.corpus import movie_reviews
# from nltk.classify import accuracy
# import random
# documents = [(list(movie_reviews.words(fileid)), category)
#     for category in movie_reviews.categories()
#     for fileid in movie_reviews.fileids(category)]
# random.shuffle(documents) # This line shuffles the order of the documents
# all_words = FreqDist(w.lower() for w in movie_reviews.words())
# word_features = list(all_words)[:2000]
# def document_features(document):
#     document_words = set(document)
#     features = {}
#     for word in word_features:
#         features['contains({})'.format(word)] = (word in document_words)
#     return features
# featuresets = [(document_features(d), c) for (d,c) in documents]
# train_set, test_set = featuresets[100:], featuresets[:100] # Split data
# classifier = NaiveBayesClassifier.train(train_set)
# print(accuracy(classifier, test_set))
print(len(sys.argv))
