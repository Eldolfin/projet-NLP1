from nltk import ngrams
from multiprocessing import Pool
import random
from utils import Prediction

Ns = [2, 3, 4]
class NgramList:
    def __init__(self, n):
        self.ngrams = n

    def add_ngram(self, n, ngram_list):
        self.ngrams[n] = ngram_list

    def count(self, ngram, n):
        return self.ngrams[n].count(ngram)

    def get_gram(self, n, i):
        if n not in self.ngrams:
            print(f"{n} not in ngrams, keys are {self.ngrams.keys()}")
            return []
        return self.ngrams[n][i]

    def __str__(self):
        return str(self.ngrams)

def train_ngrams(ds, X_train, y_train, X_test, y_test):
    print("OwO! Starting twaining for ngwams!!")

    # Decoder to translate ints to class names
    class_list = ds["train"].features["scenario"].names
    scenario_ngrams = NgramList({})
    intent_ngrams = [NgramList({})] * len(class_list)

    # Buil ngrams for each n for scenario class
    for n in Ns:
        print(".", end="")
        scenario_ngram_list = build_scenario_grams(ds, n)
        scenario_ngrams.add_ngram(n, scenario_ngram_list)

        # Build ngrams for each N for each class
        with Pool() as pool:
            # Build ngrams for each class
            results = pool.starmap(
                build_intent_grams, [(ds, n, i) for i in range(len(class_list))]
            )
        # Add ngrams to intent_ngrams
        for i in range(len(class_list)):
            intent_ngrams[i].add_ngram(n, results[i])

    print("Done!!!! (≧◡≦) \n")
    return scenario_ngrams, intent_ngrams


def ngrams_classify(
    ds, scenario_ngrams: NgramList, intent_ngrams: list, user_input: str
):
    scenario_decoder = ds["train"].features["scenario"].int2str
    intent_decoder = ds["train"].features["intent"].int2str
    class_list = ds["train"].features["scenario"].names
    intent_list = ds["train"].features["intent"].names

    # Init score vector
    scores = [0] * len(class_list)

    # Get words from input
    words = user_input.split()

    # For each ngram, check if it appears somewhere in the built scenario ngrams and if so, increase score.
    for n in Ns:
        ngram = list(ngrams(words, n))
        for t in ngram:
            for i in range(len(class_list)):
                gram = scenario_ngrams.get_gram(n, i)
                scores[i] += gram.count(t) * n

    scenario_n = scores.index(max(scores))
    scores = [0] * len(intent_list)

    # For each ngram, check if it appears somewhere in the built intent ngrams and if so, increase score.
    for n in Ns:
        ngram = list(ngrams(words, n))
        for t in ngram:
            for i in range(len(intent_list)):
                gram = intent_ngrams[scenario_n].get_gram(n, i)
                scores[i] += gram.count(t) * n

    intent_n = scores.index(max(scores))
    scenario = scenario_decoder(scenario_n)
    intent = intent_decoder(intent_n)
    
    # print(
    #    f"\nSugoi no kawaine!! Je pense que tu weux pawler de {scenario} et que tu weux plus pwecisement {intent} (≧◡≦) \n"
    # )
    
    return Prediction(
        method="ngwams",
        scenario=scenario,
        intent=intent,
        proba=scores[intent_n],
    )


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
            ngram = list(ngrams(message.lower().split(), n))
            ngram_list[i] += ngram

    return ngram_list


def build_intent_grams(ds, n, class_int):
    """
    This function builds the ngrams for the given n for the scenario class.

    returns: list of ngrams for each intent
    """
    # Isolate rows which's class name is class_int
    filtered = ds.filter(lambda x: x["scenario"] == class_int)

    scenario_ngram_list = []
    for i in range(len(ds["train"].features["intent"].names)):
        filtered = ds.filter(lambda x: x["intent"] == i)
        messageList = filtered["train"]["utt"]
        scenario_ngram_list.append([])
        for message in messageList:
            ngram = list(ngrams(message.split(), n))
            scenario_ngram_list[i] += ngram

    return scenario_ngram_list


########################### GENERATION ###########################
def ngrams_generate(
    start_word: str,
    scenario_ngrams: NgramList,
    intent_ngrams: list,
    words: int = 5,
    scenario: int = -1,
    intent: int = -1,
):
    output = start_word
    previous_words = [start_word]

    for i in range(words - 1):
        new_word = generate_next_token(
            previous_words, scenario_ngrams, intent_ngrams, scenario, intent
        )
        if new_word is None:
            return output + "?"

        output += " " + new_word
        previous_words.append(new_word)

    return output + "?"


def generate_next_token(
    previous_words: list,
    scenario_ngrams: NgramList,
    intent_ngrams: list,
    scenario: int = -1,
    intent: int = -1,
):
    """
    This function generates the next token based on the previous word and the ngrams.
    """

    for n in range(min(len(previous_words) + 1, Ns[-1]), 1, -1):
        # Build corresponding ngram from previous words
        previous_seq = tuple(previous_words[-(n - 1) :])

        if scenario > 0:
            scenario_ngram = scenario_ngrams.get_grams(n, scenario)
        else:
            scenario_ngram = [
                gram
                for scenario_grams in scenario_ngrams.ngrams[n]
                for gram in scenario_grams
            ]

        # Get next token
        if n == 2:
            filtered = list(
                filter(lambda x: x[0] == previous_seq[0], scenario_ngram)
            )  # Special case for simgle tuple
        else:
            filtered = list(
                filter(lambda x: x[: n - 1] == previous_seq, scenario_ngram)
            )

        if len(filtered) != 0:
            return random.choice(filtered)[-1]

    return None
