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
from submodels_bow import train_on_class
from phrases import UWU_PHRASES


def train_bow(ds, X_train, y_train, X_test, y_test):
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    class_name = "scenario"
    sub_class_name = "intent"

    scenario_decoder = ds["train"].features[class_name].int2str
    intent_decoder = ds["train"].features[sub_class_name].int2str

    ds = load_dataset("AmazonScience/massive", "fr-FR")
    class_name = "scenario"
    sub_class_name = "intent"
    scenario_decoder = ds["train"].features[class_name].int2str
    intent_decoder = ds["train"].features[sub_class_name].int2str

    # Access intent list
    intent_list = ds["train"].features[class_name].names
    print(intent_list)

    # model already tokenizes
    X_train = ds["train"]["utt"]
    y_train = ds["train"][class_name]

    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)

    # Train on each sub class
    class_list = ds["train"].features[class_name].names
    intent_models = {}
    for i in range(len(class_list)):
        (intent_vectorizer, intent_clf) = train_on_class(ds, i)
        intent_models[i] = (intent_vectorizer, intent_clf)

    # Train Naïve Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)

    # Test with a custom input
    test_vectorizer, test_clf = vectorizer, clf
    while True:
        user_input = input(
            "\nEntwe une fwhase à cwassifier, s'il te pwait, nya~ 💖\n> "
        )

        # Keyword to quit
        if user_input == "quit":
            return

        _, _, scenario_n = predict_scenario(user_input, vectorizer, clf)
        test_vectorizer, test_clf = intent_models[scenario_n]
        label_str = scenario_decoder(int(scenario_n))

        # Enter a phrase and print result
        (klass, proba, intent_n) = predict_scenario(
            user_input, test_vectorizer, test_clf
        )
        print(UWU_PHRASES[label_str])
        print(f"proba: {proba}")


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

    intent_number = model.classes_[klass]
    return klass, proba, intent_number
