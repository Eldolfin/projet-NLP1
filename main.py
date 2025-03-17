from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from datasets import load_dataset


def main():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    ds = load_dataset("AmazonScience/massive", "fr-FR")

    train = ds["train"].to_pandas()

    train["tokenized"] = train["utt"].apply(clean_text)
    print(train[["utt", "tokenized"]])
    nltk.bayesian(train)


def clean_text(text: str) -> List[str]:
    text = text.lower()
    words = word_tokenize(text)
    # stop_words = set(stopwords.words("french"))
    # filtered_words = [
    #     word for word in words if word.isalpha() and word not in stop_words
    #

    return words


if __name__ == "__main__":
    main()
