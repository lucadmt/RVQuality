import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def drop_series(data_frame, colname):
    return data_frame.drop(colname, axis=1)


def swap_series(full_table, s1, s2):  # swaps two series
    x = full_table[s1]
    full_table = full_table.drop(s1, axis=1)
    full_table[s1] = full_table[s2]
    full_table = full_table.drop(s2, axis=1)
    full_table[s2] = x
    return full_table


def switch_series(full_table, s1, s2):  # swaps s1, s2, deletes s2
    full_table = full_table.drop(s1, axis=1)
    full_table[s1] = full_table[s2]
    full_table = full_table.drop(s2, axis=1)
    return full_table


def drop_stopwords(wordlist):
    return list(filter(lambda y: y not in stop_words, wordlist))
