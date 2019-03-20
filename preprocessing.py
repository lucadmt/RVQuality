#!/usr/bin/env python

"""
preprocessing.py: a script which removes stopwords from
                    the root texts and saves the results
                    in another file.
"""

import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
data_frame = pd.read_csv('./dataset/reviews_filtered_20_5.csv', sep=";")
csv_text = data_frame['text']
csv_root_text = data_frame['root_text']

""" 1: bring everything to lowercase """

data_frame['text'] = csv_text.map(lambda x: x.lower())
data_frame['root_text'] = csv_root_text.map(lambda x: x.lower())

out = open("./dataset/reviews_filtered_20_5_lowercased.csv", "w")
out.write(data_frame.to_csv(sep=";"))
out.close()

""" 2: remove rows with stopwords """
