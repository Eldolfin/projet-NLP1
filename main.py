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
from submodels import train_on_class
from datetime import datetime

UWU_PHRASES = {
    "social": "UwU~ je suis ton chatbot social pweferé, toujours wà pour discuter avec twa nya~ (✿˵◕ ‿ ◕˵)",
    "transport": "Ohh, UwU! Besoin d'aide pour trouver un twain ou un bus? Je te guide avec plaisir nya~ 🚋✨",
    "calendar": "Nyaaa~ Tes wendez-vous sont tous beaux et bien rangés, veux-tu vérifier, chaton UwU? (✿˵◕ ‿ ◕˵)",
    "play": "UwU~ Prête à jouer ensemble! Que veux-tu essayer aujourd’hui, hihi? 🎮✨",
    "news": "OwO! Les dernièwes nouvelles? Je peux te les donner tout de suite, adorable lecteur UwU! 📰💖",
    "datetime": f"Nyaww~ Il est actuellement {datetime.now()} si tu te demandais, petit chat UwU! (✿˵◕ ‿ ◕˵)",
    "recommendation": "Hihi UwU~ Laisse-moi te wécommander des choses parfaites pour twa, mon chou! (≧◡≦)",
    "email": "OwO~ Un email pour toi? Je peux t'aider à le rédiger, uwu rapide comme un petit éclair ✉️✨",
    "iot": "UwU~ Je peux allumer tes wumières ou régler ta maison connectée nyah! 🌟 (✿˵◕ ‿ ◕˵)",
    "general": "Nyahaha~ Demande ce que tu veux, je suis là pour twa UwU~ 💬✨",
    "audio": "OwO~ Tu veux entendre un son ou de la musique? UwU Je suis prête! 🎵💕",
    "lists": "UwU~ Ta wiste est prête! Dis-moi ce que tu veux ajouter, hihi~ 📋✨",
    "qa": "Nyuu~ Pose-moi une qwestion, je vais tout faire pour y répondre adorablement UwU~ 💡",
    "cooking": "UwU~ On cuisine ensemble? Je trouve des recettes pour twa, hihi~ 🍳💖",
    "takeaway": "OwO~ Commande à emporter? Je peux t'aider à choisir, nyah~ 🍔✨",
    "music": "UwU~ Une chanson pour twa? Je lance ça tout de suite, hihi~ 🎶💖",
    "alarm": "Nyaa~ Réveil programmé! Je vais miauler pour twa quand il sonnera UwU~ ⏰✨",
    "weather": "UwU~ Le temps est magnifique aujourd’hui! Je te donne les détails si tu veux~ 🌤️ (✿˵◕ ‿ ◕˵)",
}


def main():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

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
    print(probabilities)
    proba = probabilities[klass]
    proba = round(proba * 100, 3)

    intent_number = model.classes_[klass]
    return klass, proba, intent_number


if __name__ == "__main__":
    main()
