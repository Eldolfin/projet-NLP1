from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression as Lr
from sklearn.svm import OneClassSVM
from typing import Tuple
from utils import Prediction


def w2v_train(
    ds,
    X_train,
    y_train,
    X_test,
    y_test,
):
    # print("Stawting w2v training... (≧◡≦) \n")

    # Load pre-trained sentence embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode the texts into vectors
    X_train_encoded = model.encode(X_train)

    # Train a classifier
    clf = Lr()
    clf.fit(X_train_encoded, y_train)

    # Train sub-models for each scenario
    intent_models = {}
    class_list = ds["train"].features["scenario"].names
    for i in range(len(class_list)):
        (intent_model, intent_clf) = train_on_class(ds, i)
        intent_models[i] = (intent_model, intent_clf)

    # print("Done!!!! (≧◡≦) \n")

    return model, clf, intent_models


def train_on_class(ds, class_index):
    """
    Trains a sub-model for a specific class.
    """
    # Filter dataset for the specific class
    filtered = ds.filter(lambda x: x["scenario"] == class_index)
    label_decoder = filtered["train"].features["scenario"].int2str

    X_train = filtered["train"]["utt"]
    y_train = filtered["train"]["intent"]
    X_test = filtered["test"]["utt"]
    y_test = filtered["test"]["intent"]

    # Encode the texts into vectors
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_encoded = model.encode(X_train)
    X_test_encoded = model.encode(X_test)

    # Train a classifier
    if len(set(filtered["train"]["intent"])) > 1:
        clf = Lr()
        clf.fit(X_train_encoded, y_train)
        score = clf.score(X_test_encoded, y_test)
    else:
        clf = OneClassSVM(gamma="auto")
        clf.fit(X_train_encoded, y_train)
        score = 1.0

    # Try testing
    label = label_decoder(class_index)

    # print(f"test dataset score for intent {label}: {score}")
    return model, clf


def w2v_classify(
    ds,
    model: SentenceTransformer,
    clf: Lr,
    intent_models: dict,
    user_input: str,
    method: str,
):
    scenario_decoder = ds["train"].features["scenario"].int2str
    intent_decoder = ds["train"].features["intent"].int2str

    text_encoded = model.encode([user_input])

    _, _, scenario_n = predict_scenario(user_input, model, clf)
    test_model, test_clf = intent_models[scenario_n]
    label_str = scenario_decoder(int(scenario_n))

    # Enter a phrase and print result
    (klass, proba, intent_n) = predict_scenario(
        user_input, test_model, test_clf
    )
    return Prediction(
        method,
        label_str,
        intent_decoder(int(intent_n)),
        proba,
    )


def predict_scenario(
    text: str, model: SentenceTransformer, clf: Lr
) -> Tuple[int, float]:
    """
    Preprocesses input text, vectorizes it, and predicts the scenario.
    returns: (class_index, confidence)
    """
    text_encoded = model.encode([text])
    probabilities = clf.predict_proba(text_encoded)[0]
    klass = probabilities.argmax()
    proba = probabilities[klass]
    proba = round(proba * 100, 3)

    intent_number = clf.classes_[klass]
    return klass, proba, intent_number
