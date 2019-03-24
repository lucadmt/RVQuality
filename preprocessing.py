#!/usr/bin/env python3

'''
preprocessing.py: a script which removes stopwords from
                    the root texts and saves the results
                    in another file.
'''

import nltk
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import reduce

from paths import lowercased_path, nostopwords_path

''' helper functions '''


def drop_stopwords(xs):
    return list(filter(lambda y: y not in stop_words, xs))


def string_to_list(str):
    return re.sub(r'^(\[\')|(\',\s\')|(\'])$', '|', str).split('|')[1:-1]


stop_words = set(stopwords.words('english'))
data_frame = pd.read_csv(lowercased_path, sep=";")

''' 1: bring everything to lowercase '''

data_frame['text'] = data_frame['text'].map(lambda x: x.lower())
data_frame['root_text'] = data_frame['root_text'].map(lambda x: x.lower())

data_frame.to_csv(lowercased_path, index=False, sep=";")

''' 2: remove rows with stopwords '''

data_frame['root_text'] = data_frame['root_text'].map(
    lambda x: string_to_list(x)
)

data_frame['text'] = data_frame['text'].map(
    lambda x: string_to_list(x)
)

data_frame['text'] = pd.Series(
    map(drop_stopwords, data_frame['text'])
)

data_frame['root_text'] = pd.Series(
    map(drop_stopwords, data_frame['root_text'])
)

data_frame.to_csv(nostopwords_path, index=False, sep=";")
