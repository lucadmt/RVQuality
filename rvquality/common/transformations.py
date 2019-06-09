import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def drop_stopwords(wordlist):
    return list(filter(lambda y: y not in stop_words, wordlist))
