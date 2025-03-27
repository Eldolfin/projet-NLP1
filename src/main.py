from bow import train_bow
from ngrams import train_ngrams
import nltk
from datasets import load_dataset


def main():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    ds = load_dataset("AmazonScience/massive", "fr-FR")
    class_name = "scenario"

    # Access intent list
    intent_list = ds["train"].features[class_name].names
    print(intent_list)

    # model already tokenizes
    X_train = ds["train"]["utt"]
    y_train = ds["train"][class_name]
    X_test = ds["test"]["utt"]
    y_test = ds["test"][class_name]

    train_ngrams(ds, X_train, y_train, X_test, y_test)

    while True:
        user_input = input(
            "\nUwU~ je suis Awexa, ton assistante pweferé!!! Toujours wà pour discuter avec twa nya~ (✿˵◕ ‿ ◕˵) \n Dis moi quel modew to souhaite entrainer  pawmi BOW|NGWAMS|NN :3 \n> "
        )

        if user_input == "BOW":
            train_bow(ds, X_train, y_train, X_test, y_test)

        elif user_input == "NGWAMS":
            print("ONII CHAN!! This is not implemented yet!!!!")
            train_ngrams(ds, X_train, y_train, X_test, y_test)

        elif user_input == "NN":
            print("ONII CHAN!! This is not implemented yet!!!!")

        elif user_input == "quit":
            return
        else:
            print("OwO I don't understand what you mean :c~")


if __name__ == "__main__":
    main()
