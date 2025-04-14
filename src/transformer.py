from sklearn.linear_model import LogisticRegression as Lr
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import OneClassSVM
from typing import Tuple
from utils import Prediction
import torch

def tf_train(
    ds,
):
    print("Stawting tf training... (≧◡≦) \n")
    
    # Load pre-trained sentence embedding model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    def encode(ds) -> dict:
        inputs = tokenizer(ds["utt"], return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return {"embedding": cls_embedding}
    
    # Encode the texts into vectors
    ds_encoded = ds.map(encode, batched=True)

    X_train = ds_encoded["train"]["embedding"]
    y_train = ds_encoded["train"]["scenario"]
    X_test = ds_encoded["test"]["embedding"]
    y_test = ds_encoded["test"]["scenario"]


    
    # Train a classifier
    clf = Lr(max_iter=1200)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    clf.fit(X_train_scaled, y_train)

    # Train sub-models for each scenario
    intent_models = {}
    class_list = ds["train"].features["scenario"].names
    for i in range(len(class_list)):
        (intent_model, intent_clf) = train_on_class(ds_encoded, i)
        intent_models[i] = (intent_model, intent_clf)

    print("Done!!!! (≧◡≦) \n")

    return model, clf, intent_models




def train_on_class(ds, class_index):
    """
    Trains a sub-model for a specific class.
    """
    # Filter dataset for the specific class
    filtered = ds.filter(lambda x: x['scenario'] == class_index)
    label_decoder = filtered["train"].features["scenario"].int2str
    
    # Encode the texts into vectors
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    X_train = filtered["train"]["embedding"]
    y_train = filtered["train"]["intent"]
    X_test = filtered["test"]["embedding"]
    y_test = filtered["test"]["intent"]
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a classifier
    if len(set(filtered["train"]["intent"])) > 1 :
        clf = Lr(max_iter=1200)
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
    else:
        clf = OneClassSVM(gamma='auto')
        clf.fit(X_train_scaled, y_train)
        score = 1.0

    # Try testing
    label = label_decoder(class_index)
    
    print(f"test dataset score for intent {label}: {score}")
    return model, clf


def tf_classify(
    ds,
    model: AutoModel,
    clf: Lr,
    intent_models: dict,
    user_input: str,
    method: str,
):
    scenario_decoder = ds["train"].features["scenario"].int2str
    intent_decoder = ds["train"].features["intent"].int2str
    

    _, _, scenario_n = predict_scenario(user_input, model, clf)
    test_model, test_clf = intent_models[scenario_n]
    label_str = scenario_decoder(int(scenario_n))

    # Enter a phrase and print result
    (klass, proba, intent_n) = predict_scenario(user_input, test_model, test_clf)
    return Prediction(
        method,
        label_str,
        intent_decoder(int(intent_n)),
        proba,
    )


def predict_scenario(
    text: str, model: AutoModel, clf: Lr
) -> Tuple[int, float]:
    """
    Preprocesses input text, vectorizes it, and predicts the scenario.
    returns: (class_index, confidence)
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] token embedding (first token)
        clf_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().reshape(1, -1)
    
    probabilities = clf.predict_proba(clf_embedding)[0]
    print(f"Probabilities: {probabilities}")
    klass = probabilities.argmax()
    proba = probabilities[klass]
    proba = round(proba * 100, 3)

    intent_number = clf.classes_[klass]
    return klass, proba, intent_number
