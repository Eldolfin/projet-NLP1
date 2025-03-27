from nltk import ngrams

NGRAMS = {}
Ns = [2, 3, 4]


def train_ngrams(ds, X_train, y_train, X_test, y_test):
    decoder = ds["train"].features["scenario"].int2str
    for n in Ns:
        build_grams(ds, n)

    while True:
        user_input = input(
            "\nEntwe une fwhase à cwassifier, s'il te pwait, nya~ 💖\n> "
        )

        # Keyword to quit
        if user_input == "quit":
            return

        scores = [0] * len(NGRAMS[n])

        words = user_input.split()
        for n in Ns:
            ngram = list(ngrams(words, n))
            for t in ngram:
                for i in range(len(NGRAMS[n])):
                    gram = NGRAMS[n][i]
                    scores[i] += gram.count(t) * n

            print(scores, decoder(scores.index(max(scores))))


def build_grams(ds, n):
    NGRAMS[n] = []
    for i in range(len(ds["train"].features["scenario"].names)):
        filtered = ds.filter(lambda x: x["scenario"] == i)
        messageList = filtered["train"]["utt"]
        NGRAMS[n].append([])
        for message in messageList:
            ngram = list(ngrams(message.split(), n))
            NGRAMS[n][i] += ngram
