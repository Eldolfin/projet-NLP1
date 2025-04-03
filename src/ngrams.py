from nltk import ngrams
from utils import ngram_list

SCENARIO_NGRAMS = ngram_list({})
INTENT_NGRAMS = []  # List de ngram_list
Ns = [2, 3, 4]


def train_ngrams(ds, X_train, y_train, X_test, y_test):
    # Decoder to translate ints to class names
    scenario_decoder = ds["train"].features["scenario"].int2str
    intent_decoder = ds["train"].features["intent"].int2str
    class_list = ds["train"].features["scenario"].names
    intent_list = ds["train"].features["intent"].names
    INTENT_NGRAMS = [ngram_list({})] * len(class_list)

    # Buil ngrams for each n for scenario class
    for n in Ns:
        build_scenario_grams(ds, n)

        # Build ngrams for each N for each class
        for i in range(len(class_list)):
            intent_ngram_list = ngram_list(build_intent_grams(ds, n, i))
            INTENT_NGRAMS[i].add_ngram(n, intent_ngram_list)

    while True:
        user_input = input(
            "\nEntwe une fwhase à cwassifier, s'il te pwait, nya~ 💖\n> "
        )

        # Keyword to quit
        if user_input == "quit":
            return

        # Init score vector
        scores = [0] * len(class_list)

        # Get words from input
        words = user_input.split()

        # For each ngram, check if it appears somewhere in the built scenario ngrams and if so, increase score.
        for n in Ns:
            ngram = list(ngrams(words, n))
            for t in ngram:
                for i in range(len(class_list)):
                    gram = SCENARIO_NGRAMS.get_gram(n, i)
                    scores[i] += gram.count(t) * n

        scenario = scores.index(max(scores))
        scores = [0] * len(intent_list)

        # For each ngram, check if it appears somewhere in the built intent ngrams and if so, increase score.
        for n in Ns:
            ngram = list(ngrams(words, n))
            for t in ngram:
                for i in range(len(intent_list)):
                    gram = INTENT_NGRAMS[scenario].get_gram(n, i)
                    scores[i] += gram.count(t) * n

        intent = scores.index(max(scores))
        print(scores, scenario_decoder(scenario), intent_decoder(intent))


def build_scenario_grams(ds, n):
    """
    This function builds the ngrams for the given n for the scenario class.
    """
    ngram_list = []
    for i in range(len(ds["train"].features["scenario"].names)):
        filtered = ds.filter(lambda x: x["scenario"] == i)
        messageList = filtered["train"]["utt"]
        ngram_list.append([])
        for message in messageList:
            ngram = list(ngrams(message.split(), n))
            ngram_list[i] += ngram

    SCENARIO_NGRAMS.add_ngram(n, ngram_list)


def build_intent_grams(ds, n, class_int):
    """
    This function builds the ngrams for the given n for the scenario class.

    returns: list of ngrams for each intent
    """
    # Isolate rows which's class name is class_int
    filtered = ds.filter(lambda x: x["scenario"] == class_int)

    ngram_list = []
    for i in range(len(ds["train"].features["intent"].names)):
        filtered = ds.filter(lambda x: x["intent"] == i)
        messageList = filtered["train"]["utt"]
        ngram_list.append([])
        for message in messageList:
            ngram = list(ngrams(message.split(), n))
            ngram_list[i] += ngram

    return ngram_list
