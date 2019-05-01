#!/usr/bin/env python3
import os
import ast
import sys
import csv
import numpy as np
import pandas as pd
from thesis.common.transformations import drop_stopwords, string_to_list
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce
from string import punctuation

script, inputfile, outputfile = sys.argv

if not os.path.isfile(inputfile):
    sys.exit("Error: No such file: "+inputfile)

data_frame = pd.read_csv(inputfile, sep=';')

if 'comments' not in data_frame.columns:
    sys.exit("Error: the file you provided hasn't 'comments' column")

# split to a list of lowercased words, then drop stopwords
data_frame['comments'] = data_frame['comments'].map(
    lambda x: x.lower().translate(str.maketrans('', '', punctuation+'\n\r\xa0\xad')))
# data_frame['comments'] = pd.Series(map(drop_stopwords, data_frame['comments']))

# lemmatize + reduce
lemmatizer = WordNetLemmatizer()
data_frame['lemmas'] = data_frame['comments'].map(
    lambda words: reduce(
        lambda x, y: x + " " + y,  # concatenate all words
        map(lambda word: lemmatizer.lemmatize(word), words),
        ""
    )
)

vectorizer = TfidfVectorizer(
    strip_accents='unicode',
    smooth_idf=True,
    stop_words='english',
    ngram_range=(1, 1)
)

tfidf_matrix = vectorizer.fit_transform([x for x in data_frame['comments']])
mcoo = tfidf_matrix.tocoo()
tfidf_dict = {k: v for k, v in zip(vectorizer.get_feature_names(), mcoo.data)}

of = open(outputfile.replace('.csv', '.dict'), 'w')
of.write(str(tfidf_dict))
of.close()

uncaptured = set()


def get_vect(comment, tf_idfs_dict):
    tf_idf_vect = []
    for x in tf_idfs_dict:
        try:
            if x in comment:
                tf_idf_vect.append(tf_idfs_dict[x])
        except KeyError as ke:
            print(x)
            uncaptured.add(x)
        pass
    return tf_idf_vect


data_frame['tf_idf_vect'] = data_frame['comments'].map(
    lambda comment: get_vect(comment, tfidf_dict)
)

print(uncaptured)

of = open(outputfile.replace('.csv', '.set').replace(
    'tf_idfs', 'uncaptured').replace('out', 'uncaptured'), 'w')
of.write(str(uncaptured))
of.close()

data_frame['tf_idf_mean'] = data_frame['tf_idf_vect'].map(
    lambda row: float(sum(row)) / max(len(row), 1)
)

data_frame['tf_idf_sum'] = data_frame['tf_idf_vect'].map(sum)

data_frame.to_csv(outputfile, index=False,
                  sep=";", quoting=csv.QUOTE_ALL)
