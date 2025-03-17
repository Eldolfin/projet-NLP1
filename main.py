from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from datasets import load_dataset


def main():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    ds = load_dataset("AmazonScience/massive", "fr-FR")
    class_name = "scenario"
    label_decoder = ds["train"].features[class_name].int2str

    # model already tokenizes
    # train["tokenized"] = train["utt"].apply(clean_text)
    X_train = ds["train"]["utt"]
    y_train = ds["train"][class_name]
    X_test = ds["train"]["utt"]
    y_test = ds["train"][class_name]

    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.fit_transform(X_test)

    # Train Naïve Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)
    score = clf.score(X_test_bow, y_test)
    print(f"test dataset score: {score}")

    # phrase = input("Que veux tu que je fasse? :\n>")
    # phrase = ["Allume la lumiere"]
    # clf.predict(vectorizer.fit_transform(phrase))

    # Test with a custom input
    while True:
        test_phrase = input("\nEnter a phrase to classify: ")
        (label, proba) = predict_scenario(test_phrase, vectorizer, clf)
        label = label_decoder(int(label))
        print(f"Ok je fais {label} (sur a {proba}%)")


def predict_scenario(
    text: str, vectorizer: CountVectorizer, model: MultinomialNB
) -> Tuple[int, float]:
    """
    Preprocesses input text, vectorizes it, and predicts the scenario.
    returns: (class_index, confidence)
    """
    text_bow = vectorizer.transform([text])  # Convert to BoW
    probabilities = model.predict_proba(text_bow)[0]
    klass = probabilities.argmax()
    proba = probabilities[klass]
    proba = round(proba * 100, 3)
    return klass, proba


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
