#!/usr/bin/env python3

"""
preprocessing.py: a script which removes stopwords from
                    the root texts and saves the results
                    in another file.
"""

import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import reduce

""" helper functions """


def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def drop_void(xs):
    return xs.map(lambda x: (x, None)[x == []]).dropna()


def drop_stopwords(xs):
    return xs.map(
        lambda x: (x, None)[None in map(
            lambda y: (y, None)[y in stop_words], x
        )]
    ).dropna()


def stringlist_to_tokenlist(str):
    return word_tokenize(str.replace('[', '')
                            .replace(']', '')
                            .replace("'", '')
                            .replace(',', ''))

stop_words = set(stopwords.words('english'))
data_frame = pd.read_csv('./dataset/reviews_filtered_20_5.csv', sep=";")

""" 1: bring everything to lowercase """

data_frame['text'] = data_frame['text'].map(lambda x: x.lower())
data_frame['root_text'] = data_frame['root_text'].map(lambda x: x.lower())

data_frame.to_csv("./dataset/reviews_filtered_20_5_lowercased.csv", index=False, sep=";")

""" 2: remove rows with stopwords """

csv_text = data_frame['text']
csv_root_text = data_frame['root_text']

csv_root_text = csv_root_text.map(lambda x: stringlist_to_tokenlist(x))
csv_text = csv_text.map(lambda x: stringlist_to_tokenlist(x))

clear = compose(drop_stopwords, drop_void)

data_frame['text'] = clear(csv_text)
data_frame['root_text'] = clear(csv_root_text)
data_frame.dropna()

data_frame.to_csv("./dataset/reviews_filtered_20_5_lowercased_nostopwords.csv", index=False, sep=";")
