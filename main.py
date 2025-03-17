from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from datasets import load_dataset


def main():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    ds = load_dataset("AmazonScience/massive", "fr-FR")

    # model already tokenizes
    # train["tokenized"] = train["utt"].apply(clean_text)
    X_train = ds["train"]["utt"]
    y_train = ds["train"]["scenario"]
    X_test = ds["train"]["utt"]
    y_test = ds["train"]["scenario"]

    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.fit_transform(X_test)

    # Train Naïve Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)
    score = clf.score(X_test_bow, y_test)

    print(score)


def clean_text(text: str) -> List[str]:
    text = text.lower()
    words = word_tokenize(text)
    # stop_words = set(stopwords.words("french"))
    # filtered_words = [
    #     word for word in words if word.isalpha() and word not in stop_words
    #

    return words


if __name__ == "__main__":
    main()
