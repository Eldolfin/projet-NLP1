class ngram_list:
    def __init__(self, n):
        self.ngrams = n

    def add_ngram(self, n, ngram_list):
        self.ngrams[n] = ngram_list

    def count(self, ngram, n):
        return self.ngrams[n].count(ngram)

    def get_gram(self, n, i):
        return self.ngrams[n][i]

    def __str__(self):
        return str(self.ngrams)
