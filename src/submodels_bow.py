from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression as Lr

from datasets import Dataset


def train_on_class(
    ds: Dataset, class_int: int
) -> Tuple[CountVectorizer, Lr]:
    # Isolate rows which's class name is class_int
    filtered = ds.filter(lambda x: x["scenario"] == class_int)
    label_decoder = filtered["train"].features["scenario"].int2str

    # Create train and test sets
    X_train = filtered["train"]["utt"]
    y_train = filtered["train"]["intent"]
    X_test = filtered["test"]["utt"]
    y_test = filtered["test"]["intent"]

    vectorizer = TfidfVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train Naïve Bayes classifier
    if len(set(filtered["train"]["intent"])) > 1 :
        clf = Lr()
    else:
        clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)
    score = clf.score(X_test_bow, y_test)

    label = label_decoder(class_int)
    print(f"test dataset score for intent {label}: {score}")
    return vectorizer, clf
