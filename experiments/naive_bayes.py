from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

from datasets import load_dataset


def main():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    ds = load_dataset("AmazonScience/massive", "fr-FR")
    class_name = "scenario"
    label_decoder = ds["train"].features[class_name].int2str

    labels = ds["train"].features[class_name].names

    # model already tokenizes
    # train["tokenized"] = train["utt"].apply(clean_text)
    X_train = ds["train"]["utt"]
    y_train = ds["train"][class_name]
    X_test = ds["test"]["utt"]
    y_test = ds["test"][class_name]

    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train Naïve Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)

    #Once trained, indicate scoring metrics with test dataset predictions
    predictions = clf.predict(X_test_bow)
    print(metrics.classification_report(y_test, predictions, target_names=labels))

    # phrase = input("Que veux tu que je fasse? :\n>")
    # phrase = ["Allume la lumiere"]
    # clf.predict(vectorizer.fit_transform(phrase))

    # Test with a custom input
    


if __name__ == "__main__":
    main()
