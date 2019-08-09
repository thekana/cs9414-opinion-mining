# ------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#
# ------------------------------------------------------------------------------+
from __future__ import division
# --- IMPORT DEPENDENCIES ------------------------------------------------------+
import csv
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import metrics, tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import random
import math

# --- COST FUNCTION ------------------------------------------------------------+
# function we are attempting to optimize (minimize)


def func1(x):
    total = 0
    for i in range(len(x)):
        total += x[i]**2
    return total

# --- MAIN ---------------------------------------------------------------------+


class Particle:
    def __init__(self, x0):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.err_i = -1               # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        # constant inertia weight (how much to weigh the previous velocity)
        w = 0.5
        c1 = 1        # cognative constant
        c2 = 2        # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social = c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i] = w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            # print(self.position_i[i])
            # print(self.velocity_i[i])
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO():
    def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
        global num_dimensions
        num_dimensions = len(x0)
        err_best_g = -1                   # best error for group
        pos_best_g = []                   # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            if i > num_particles/2:
                x = [random.randint(x0[0], bounds[0][1]), random.randint(
                    x0[1], bounds[1][1]), random.randint(x0[2], bounds[2][1]), random.random()]
            else:
                x = [random.randint(bounds[0][0], x0[0]), random.randint(
                    bounds[1][0], x0[1]), random.randint(bounds[2][0], x0[2]), random.random()]
            swarm.append(Particle(x))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            print("Iteration:", i)
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)
        print('Accuracy:', 1-err_best_g)


# --- RUN ----------------------------------------------------------------------+
df = pd.read_csv('dataset.tsv', sep='\t', quoting=csv.QUOTE_NONE, dtype=str, encoding='utf-8',
                 header=None, names=["instance", "text", "id", "sentiment", "is_sarcastic"])

###
""" Functions for text pre-processing """


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", " ", sample)


def remove_punctuation(sample):
    """Remove punctuations from a sample string"""
    return re.sub(r'[^\w\s\&\#\@\$\%\_]', '', sample)


def myTokenizer(sample):
    """Customized tokenizer"""
    new_words = []
    words = sample.split(' ')
    new_words = [word for word in words if len(
        word) >= 2 and not word.startswith('au') and not word.startswith('#aus')]
    return new_words


def remove_stopwords_NLTK(sample):
    """Remove stopwords using NLTK"""
    stopWords = set(stopwords.words('english'))
    words = [w for w in sample.split(' ') if len(w) >= 2]
    filteredText = ""
    for word in words:
        if word not in stopWords:
            filteredText = filteredText + word + " "
    return filteredText.rstrip()


def remove_digits(input_text):
    return re.sub('\d+', '', input_text)


def porter_stem(sample):
    """Stemming"""
    words = [w for w in sample.split(' ') if len(w) >= 2]
    ps = PorterStemmer()
    stemmed_text = ""
    for word in words:
        stemmed_text = stemmed_text + ps.stem(word) + " "
    return stemmed_text.rstrip()


def myPreprocessor(sample):
    """Customized preprocessor"""
    sample = remove_URL(sample)
    sample = sample.lower()
    sample = remove_stopwords_NLTK(sample)
    sample = remove_punctuation(sample)
    sample = remove_digits(sample)
    sample = porter_stem(sample)
    return sample


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
X_train_, X_test_, y_train, y_test = train_test_split(
    text_data, Y, test_size=0.25, shuffle=False)


def classifierCost(n):

    print("Trying: ", n[0], n[1], n[2], n[3])
    count = CountVectorizer(preprocessor=myPreprocessor, tokenizer=myTokenizer,
                            max_features=round(n[0]), ngram_range=(1, round(n[1])), min_df=round(n[2]))
    # count = TfidfVectorizer(preprocessor=myPreprocessor, tokenizer=myTokenizer,
    #                         max_features=round(n[0]), ngram_range=(1, round(n[1])), min_df=round(n[2]), use_idf=True)
    X_train = count.fit_transform(X_train_).toarray()
    X_test = count.transform(X_test_).toarray()
    clf = MultinomialNB(alpha=n[3], fit_prior=True)
    #clf = BernoulliNB(alpha=1.0, fit_prior=True)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    y_pred = model.predict(X_train)
    report1 = classification_report(y_train, y_pred, output_dict=True)
    return 1 - 2*(report['micro avg']['f1-score']*report1['micro avg']['f1-score'])/(report['micro avg']['f1-score']+report1['micro avg']['f1-score'])


# initial starting location [x1,x2...]
initial = [400, 1, 5, 0.5]
# input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
bounds = [(100, 1500), (1, 3), (0, 5), (0.3, 1)]
PSO(classifierCost, initial, bounds, num_particles=20, maxiter=10)
