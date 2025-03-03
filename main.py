from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
import re
from typing import List

nltk.download("punkt_tab")
nltk.download("stopwords")


# DATASET = "data/training.1600000.processed.noemoticon.csv"
DATASET = "data/small.csv"


def main():
    df = pd.read_csv(
        DATASET,
        encoding="ISO-8859-1",
        names=["target", "ids", "date", "flag", "user", "text"],
    )
    df["cleaned_text"] = df["text"].apply(clean_text)
    print(df["cleaned_text"])
    # TODO: compute probabilies


def classify(text: str) -> int:
    """
    returns: int between 0-4: 0 = negative, 2 = neutral, 4 = positive
    """
    # TODO
    return -1


def clean_text(text: str) -> List[str]:
    text = text.lower()
    # text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word.isalpha() and word not in stop_words
    ]

    return filtered_words


if __name__ == "__main__":
    main()
