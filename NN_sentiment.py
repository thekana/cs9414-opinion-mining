import pandas as pd
import csv
import sys
import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree, metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

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

training_size = 1800

count = CountVectorizer(preprocessor=myPreprocessor,
                        lowercase=False, tokenizer=myTokenizer, max_features=None)
bag_of_words = count.fit_transform(text_data)
# print(count.get_feature_names())
# print(count.vocabulary_)
X = bag_of_words.toarray()
# creating target classes
Y = np.array([])
for text in df.sentiment:
    Y = np.append(Y, text)
# First 1500 for training set, last 500 for test set
x_train = np.asarray(X[:training_size])
x_test = np.asarray(X[training_size:])


# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(Y[:training_size])
y_train = encoder.transform(Y[:training_size])
y_test = encoder.transform(Y[training_size:])

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

batch_size = 5
epochs = 3

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(6907,), activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(3,activation=tf.nn.softmax))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
