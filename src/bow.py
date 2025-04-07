from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer as Cv, TfidfVectorizer as Tfv
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as Lr
from submodels_bow import train_on_class
from phrases import UWU_PHRASES
from utils import Prediction


def train_bow(ds, X_train, y_train, X_test, y_test):
    # Access intent list
    intent_list = ds["train"].features["scenario"].names
    print("UwU! Starting training for BoW!!...\n")

    vectorizer = Tfv()
    X_train_bow = vectorizer.fit_transform(X_train)

    # Train on each sub class
    intent_models = {}
    class_list = ds["train"].features["scenario"].names
    for i in range(len(class_list)):
        (intent_vectorizer, intent_clf) = train_on_class(ds, i)
        intent_models[i] = (intent_vectorizer, intent_clf)

    # Train Naïve Bayes classifier
    clf = Lr()
    clf.fit(X_train_bow, y_train)

    print("Done!!!! (≧◡≦) \n")
    return vectorizer, clf, intent_models


def bow_classify(
    ds,
    vectorizer: Tfv,
    clf: Lr,
    intent_models: dict,
    user_input,
):
    scenario_decoder = ds["train"].features["scenario"].int2str
    intent_decoder = ds["train"].features["intent"].int2str

    _, _, scenario_n = predict_scenario(user_input, vectorizer, clf)
    test_vectorizer, test_clf = intent_models[scenario_n]
    label_str = scenario_decoder(int(scenario_n))

    # Enter a phrase and print result
    (klass, proba, intent_n) = predict_scenario(user_input, test_vectorizer, test_clf)
    return Prediction(
        "BoW",
        label_str,
        intent_decoder(int(intent_n)),
        proba,
    )


def predict_scenario(
    text: str, vectorizer: Tfv, clf: Lr
) -> Tuple[int, float]:
    """
    Preprocesses input text, vectorizes it, and predicts the scenario.
    returns: (class_index, confidence)
    """
    text_bow = vectorizer.transform([text])  # Convert to BoW
    probabilities = clf.predict_proba(text_bow)[0]
    klass = probabilities.argmax()
    proba = probabilities[klass]
    proba = round(proba * 100, 3)

    intent_number = clf.classes_[klass]
    return klass, proba, intent_number
