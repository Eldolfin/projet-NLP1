from sklearn.linear_model import LogisticRegression as Lr, SGDClassifier as SGD
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DistilBertForSequenceClassification,
)
from sklearn import preprocessing
from typing import Tuple
from utils import Prediction
from datasets import Dataset, Value
import torch
import time

training_args = TrainingArguments(
    output_dir="transformer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)


def tf_train(
    ds,
    X_train,
    y_train,
    X_test,
    y_test,
):
    # Define toenizer and model to use
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(ds["train"].features["scenario"].names),
    )
    X_train_ds = Dataset.from_dict({"text": X_train, "labels": y_train})
    X_test_ds = Dataset.from_dict({"text": X_test, "labels": y_test})

    # tokenize the inputs
    def preprocess(input):
        result = tokenizer(input["text"], padding="max_length", truncation=True)
        result["labels"] = input["labels"]
        return result

    X_tokenized_train = X_train_ds.map(preprocess, batched=True).cast_column(
        "labels", Value("int64")
    )
    X_tokenized_test = X_test_ds.map(preprocess, batched=True).cast_column(
        "labels", Value("int64")
    )

    # Define the actual transformer above the model and train it
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=X_tokenized_train,
        eval_dataset=X_tokenized_test,
    )
    trainer.train()

    # Evaluate the model
    trainer.evaluate()
    trainer.predict(X_test, y_test)

    # Train sub-models for each scenario
    intent_models = {}
    class_list = ds["train"].features["scenario"].names
    # for i in range(len(class_list)):
    #     (intent_model, intent_clf) = train_on_class(ds_encoded, i)
    #     intent_models[i] = (intent_model, intent_clf)

    return model, trainer, intent_models


def train_on_class(ds, class_index):
    """
    Trains a sub-model for a specific class.
    """
    # Filter dataset for the specific class
    filtered = ds.filter(lambda x: x["scenario"] == class_index)
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
    if len(set(filtered["train"]["intent"])) > 1:
        clf = Lr(max_iter=1200)
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
    else:
        clf = SGD(max_iter=1200)
        clf.fit(X_train_scaled, y_train)
        score = 1.0

    # Try testing
    label = label_decoder(class_index)

    return model, clf


def tf_classify(
    ds,
    model: AutoModel,
    clf: Lr,
    intent_models: dict,
    user_input: str,
    method: str,
):
    before = time.process_time()
    scenario_decoder = ds["train"].features["scenario"].int2str
    intent_decoder = ds["train"].features["intent"].int2str

    _, _, scenario_n = predict_scenario(user_input, model, clf)
    # test_model, test_clf = intent_models[scenario_n]
    label_str = scenario_decoder(int(scenario_n))

    # Enter a phrase and print result
    (klass, proba, intent_n) = (
        "unknown",
        0,
        0,
    )  # predict_scenario(user_input, test_model, test_clf)
    return Prediction(
        method,
        label_str,
        intent_decoder(int(intent_n)),
        proba,
        before=before,
    )


def predict_scenario(text: str, model: AutoModel, clf: Lr) -> Tuple[int, float]:
    """
    Preprocesses input text, vectorizes it, and predicts the scenario.
    returns: (class_index, confidence)
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] token embedding (first token)
        clf_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().reshape(1, -1)

    probabilities = clf.predict_proba(clf_embedding)[0]
    print(probabilities)

    klass = probabilities.argmax()
    proba = probabilities[klass]
    proba = round(proba * 100, 3)

    intent_number = clf.classes_[klass]
    return klass, proba, intent_number
