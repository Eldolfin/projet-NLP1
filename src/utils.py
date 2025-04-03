class ngram_list:
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
