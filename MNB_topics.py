from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])

# Perform shuffle
# df = shuffle(df)
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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(
        y_true.astype('int'), y_pred.astype('int'))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


class_names = np.asarray(list(range(0, 20)))

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
for text in df.id:
    Y = np.append(Y, int(text)-10000)
# First 1500 for training set, last 500 for test set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, shuffle=False)

start_time = time.time()
clf = MultinomialNB()
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
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                      title='Confusion matrix, without normalization')
test_report = classification_report(y_test, y_pred, output_dict=True)

start_time = time.time()
y_pred = model.predict(X_train)
# print(classification_report(y_train, y_pred))
# print('Accuracy score:', accuracy_score(y_train, y_pred))
trainingtime = (time.time() - start_time + training_time)


train_report = classification_report(y_train, y_pred, output_dict=True)

metric_list = ['precision', 'recall', 'f1-score']
avg_list = ['micro avg', 'macro avg', 'weighted avg']

test_str_output = "MNB_topics\t"+f"{size}\t"+"test\t"
train_str_output = "MNB_topics\t"+f"{size}\t"+"train\t"

for m in metric_list:
    for a in avg_list:
        test_str_output = test_str_output + f"{test_report[a][m]:.3f}\t"
        train_str_output = train_str_output + f"{train_report[a][m]:.3f}\t"
test_str_output += f"{testtime:.4f}"
train_str_output += f"{trainingtime:.4f}"
print(test_str_output.rstrip())
print(train_str_output.rstrip())


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix


# Plot normalized confusion matrix
# plot_confusion_matrix(y_train, y_pred, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()
