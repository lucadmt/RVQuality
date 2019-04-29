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


def display_scores_tfidf(vectorizer, tfidf_result):
    dict_str = ""
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    dict_str = "{"
    for item in sorted_scores:
        dict_str += "'{}': {},".format(item[0], item[1])
    dict_str += "}"
    return ast.literal_eval(dict_str)


script, inputfile, outputfile = sys.argv

if not os.path.isfile(inputfile):
    sys.exit("Error: No such file: "+inputfile)

data_frame = pd.read_csv(inputfile, sep=';')

if 'comments' not in data_frame.columns:
    sys.exit("Error: the file you provided hasn't 'comments' column")

# split to a list of lowercased words, then drop stopwords
data_frame['comments'] = data_frame['comments'].map(
    lambda x: x.lower().split(' '))
data_frame['comments'] = pd.Series(map(drop_stopwords, data_frame['comments']))

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
    smooth_idf=True,
    stop_words='english',
    ngram_range=(1, 1)
)
tfidf_matrix = vectorizer.fit_transform(data_frame['lemmas'])
tf_idf_dict = display_scores_tfidf(vectorizer, tfidf_matrix)

of = open(outputfile.replace('.csv', '.dict'), 'w')
of.write(str(tf_idf_dict))
of.close()

data_frame['tf_idf_vect'] = data_frame['lemmas'].map(
    lambda ll: [tf_idf_dict[x] for x in ll.split(' ') if x in tf_idf_dict]
)

data_frame['tf_idf_mean'] = data_frame['tf_idf_vect'].map(
    lambda row: float(sum(row)) / max(len(row), 1)
)

data_frame['tf_idf_sum'] = data_frame['tf_idf_vect'].map(sum)

data_frame.to_csv(outputfile, index=False,
                  sep=";", quoting=csv.QUOTE_ALL)
