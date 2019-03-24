#!/usr/bin/env python3

import pandas as pd
import numpy as np

from preprocessing import string_to_list


def build_reviewer_stats(rv_data_frame):
    # substitute comments with a number representing # of words in it
    rv_data_frame['comments'] = \
        rv_data_frame['comments'].map(lambda x: len(x.split(' ')))
    reviewers = rv_data_frame.groupby(['reviewer_id'])

    # remove duplicates from root_text lists
    rv_data_frame['root_text'] = \
        rv_data_frame['root_text'].map(lambda x: len(list(dict.fromkeys(x))))

    return pd.DataFrame(data={
            "num_review": reviewers.reviewer_id.count(),
            "min_word_count": reviewers.comments.min(),
            "mean_word_count": reviewers.comments.mean(),
            "max_word_count": reviewers.comments.max(),
            "median_word_count": reviewers.comments.median(),
            "min_root_count": reviewers.root_text.min(),
            "mean_root_count": reviewers.root_text.mean(),
            "max_root_count": reviewers.root_text.max(),
            "median_root_count": reviewers.root_text.median()
    })

nostopwords_path = "./dataset/reviews_filtered_20_5_lowercased_nostopwords.csv"
reviewers_path = "./dataset/reviews_filtered_20_5_rstat.csv"
data_frame = pd.read_csv(nostopwords_path, sep=";")

data_frame['root_text'] = data_frame['root_text'].map(
    lambda x: string_to_list(x)
)

build_reviewer_stats(data_frame).to_csv(reviewers_path, sep=";")
