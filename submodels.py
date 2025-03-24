from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def train_on_class(ds, class_int):
    # Isolate rows which's class name is class_int
    ds = ds.filter(lambda x: x["scenario"] == class_int)
    label_decoder = ds["train"].features["scenario"].int2str

    # Xreate train and test sets
    X_train = ds["train"]["utt"]
    y_train = ds["train"]["intent"]
    X_test = ds["test"]["utt"]
    y_test = ds["test"]["intent"]

    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train Naïve Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)
    score = clf.score(X_test_bow, y_test)

    label = label_decoder(int(class_int))
    print(f"test dataset score for intent {label}: {score}")
    return vectorizer, clf
