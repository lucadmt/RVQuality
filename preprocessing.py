#!/usr/bin/env python3

'''
preprocessing.py: a script which removes stopwords from
                    the root texts and saves the results
                    in another file.
'''

import re
import csv
import pandas as pd

from nltk.corpus import stopwords
from functools import reduce

from paths import lowercased_full_path, nostopwords_full_path, main_full_path
from thesis.common.transformations import drop_stopwords, string_to_list

data_frame = pd.read_csv(main_full_path, sep=";")

''' 1: bring everything to lowercase '''

data_frame['text'] = data_frame['text'].map(lambda x: x.lower())
data_frame['root_text'] = data_frame['root_text'].map(lambda x: x.lower())

data_frame.to_csv(lowercased_full_path, index=False,
                  sep=";", quoting=csv.QUOTE_ALL)

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

data_frame.to_csv(nostopwords_full_path, index=False,
                  sep=";", quoting=csv.QUOTE_ALL)
