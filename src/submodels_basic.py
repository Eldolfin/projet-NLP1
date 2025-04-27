from typing import Tuple
from sklearn.naive_bayes import MultinomialNB

from datasets import Dataset


def train_on_class(
    ds: Dataset, class_int: int, vectorizer_template: type, clf_template: type
) -> Tuple:
    # Isolate rows which's class name is class_int
    filtered = ds.filter(lambda x: x["scenario"] == class_int)
    label_decoder = filtered["train"].features["scenario"].int2str

    # Create train and test sets
    X_train = filtered["train"]["utt"]
    y_train = filtered["train"]["intent"]
    X_test = filtered["test"]["utt"]
    y_test = filtered["test"]["intent"]

    vectorizer = vectorizer_template()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train Naïve Bayes classifier
    if len(set(filtered["train"]["intent"])) > 1:
        clf = clf_template()
    else:
        clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)
    score = clf.score(X_test_bow, y_test)

    label = label_decoder(class_int)
    return vectorizer, clf
