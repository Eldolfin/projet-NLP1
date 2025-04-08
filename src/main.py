from basic import basic_train, basic_classify
from ngrams import train_ngrams, ngrams_classify, ngrams_generate
from word2vec import w2v_train, w2v_classify
import nltk
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer as Tfv
from sklearn.linear_model import LogisticRegression as Lr
from sklearn.naive_bayes import MultinomialNB as Mnb


def main():
    print("Downloading ntlk...\n")
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    print("Done!\n")

    ds = load_dataset("AmazonScience/massive", "fr-FR")
    class_name = "scenario"

    # model already tokenizes
    X_train = ds["train"]["utt"]
    y_train = ds["train"][class_name]
    X_test = ds["test"]["utt"]
    y_test = ds["test"][class_name]

    print(
        "\nUwU~ je suis Awexa, ton assistante pweferé!!! Toujours wà pour discuter avec twa nya~ (✿˵◕ ‿ ◕˵) \n\n"
    )

    # Train models
    # bow_vectorizer, bow_clf, bow_intent_models = basic_train(ds, X_train, y_train, X_test, y_test, CountVectorizer, Mnb)
    # idf_vectorizer, idf_clf, idf_intent_models = basic_train(ds, X_train, y_train, X_test, y_test, Tfv, Lr)
    w2v_model, w2v_clf, w2v_intent_models = w2v_train(ds, X_train, y_train, X_test, y_test)
    # scenario_grams, intent_grams = train_ngrams(ds, X_train, y_train, X_test, y_test)

    # for i in range(10):
        # print(ngrams_generate("quelle", scenario_grams, intent_grams, 15))

    while True:
        user_input = input(
            "\nEntwe une fwhase à cwassifier, s'il te pwait, nya~ 💖\n> "
        ).lower()

        if user_input == "quit":
            return
        
        print("")
        # print(basic_classify(ds, bow_vectorizer, bow_clf, bow_intent_models, user_input, "bow"))
        # print(basic_classify(ds, idf_vectorizer, idf_clf, idf_intent_models, user_input, "idf"))
        print(w2v_classify(ds, w2v_model, w2v_clf, w2v_intent_models, user_input, "word2vec"))
        # print(ngrams_classify(ds, scenario_grams, intent_grams, user_input))

        # TODO: print this with the consensus (majority of votes between models) 
        # f"Sugoi no kawaine!!\n D'apwes la method {self.method}, we pense que tu weux pawler de {self.scenario} et que tu weux plus pwecisement {self.intent} awec une pwoba de {self.proba} (≧◡≦) \n"

        # TODO: add train_nn


if __name__ == "__main__":
    main()
