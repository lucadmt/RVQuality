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
    lambda x: x.lower().translate(
        str.maketrans('', '', punctuation+'\n\r\xa0\xad')
    ).split(' ')
)

data_frame['comments'] = pd.Series(
    map(drop_stopwords, data_frame['comments'])
)

# lemmatize
lemmatizer = WordNetLemmatizer()
data_frame['lemmas'] = data_frame['comments'].map(
    lambda words:
    list(map(lambda word: lemmatizer.lemmatize(word), words))
)

# remove empty string
data_frame['lemmas'] = data_frame['lemmas'].map(
    lambda list_row:
    list(
        filter(
            lambda word: word is not '', list_row
        )
    )
)

# build the corpus
string_lemmas = data_frame['lemmas'].map(lambda list_row: " ".join(list_row))
corpus = [x for x in string_lemmas]

# let any word be considered
vectorizer = TfidfVectorizer(
    smooth_idf=True,
    ngram_range=(1, 1),
    token_pattern=r'\S{3,}'
)

sparse = vectorizer.fit_transform(corpus)


def get_tf_idf_vect(tf_idf_matrix, lemma_list, idx):
    return list(map(lambda word: tf_idf_matrix[word][idx], lemma_list))


# convert the sparse matrix to dataframe
cols = vectorizer.get_feature_names()
dense = sparse.todense()
tf_idf_df = pd.DataFrame(dense, columns=cols)

lemmas = data_frame['lemmas']

data_frame['tf_idf_vect'] = pd.Series(
    [
        get_tf_idf_vect(tf_idf_df, lemmas[idx], idx)
        for idx in range(0, len(data_frame.values))
    ]
)

data_frame['tf_idf_mean'] = data_frame['tf_idf_vect'].map(
    lambda row: float(sum(row)) / max(len(row), 1)
)

data_frame['tf_idf_sum'] = data_frame['tf_idf_vect'].map(sum)

data_frame.to_csv(outputfile, index=False,
                  sep=";", quoting=csv.QUOTE_ALL)
