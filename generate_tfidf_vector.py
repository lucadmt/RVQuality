#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from thesis.common.transformations import drop_stopwords, string_to_list
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce


def display_scores_tfidf(vectorizer, tfidf_result, out_file):
    f = open(out_file, "w")
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        f.write("{0:50} Score: {1}".format(item[0], item[1]))
        f.write("\n")
    f.close()


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
display_scores_tfidf(vectorizer, tfidf_matrix, outputfile)
