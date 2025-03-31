from bow import train_bow
import nltk
from datasets import load_dataset


def main():
    print("Downloading ntlk...\n")
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    ds = load_dataset("AmazonScience/massive", "fr-FR")
    class_name = "scenario"

    # model already tokenizes
    X_train = ds["train"]["utt"]
    y_train = ds["train"][class_name]
    X_test = ds["test"]["utt"]
    y_test = ds["test"][class_name]

    print("\nUwU~ je suis Awexa, ton assistante pweferé!!! Toujours wà pour discuter avec twa nya~ (✿˵◕ ‿ ◕˵) \n\n")

    while True:

        user_input = input(
            "\nEntwe une fwhase à cwassifier, s'il te pwait, nya~ 💖\n> "
        )

        if user_input == "quit":
            return

        train_bow(ds, X_train, y_train, X_test, y_test, user_input)

        # TODO: add user_input argument in train_ngrams function
        # train_ngrams(ds, X_train, y_train, X_test, y_test)

        # TODO: add train_nn


if __name__ == "__main__":
    main()
