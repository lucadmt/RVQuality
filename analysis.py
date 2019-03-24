#!/usr/bin/env python3

import pandas as pd

from preprocessing import string_to_list
from paths import nostopwords_path, reviewers_stats_path, listings_stats_path


def build_reviewer_stats(reviewers):
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


def build_listings_stats(listings):
    return pd.DataFrame(data={
            "num_review": listings.listing_id.count(),
            "min_word_count": listings.comments.min(),
            "mean_word_count": listings.comments.mean(),
            "max_word_count": listings.comments.max(),
            "median_word_count": listings.comments.median(),
            "min_root_count": listings.root_text.min(),
            "mean_root_count": listings.root_text.mean(),
            "max_root_count": listings.root_text.max(),
            "median_root_count": listings.root_text.median()
    })

data_frame = pd.read_csv(nostopwords_path, sep=";")

data_frame['root_text'] = data_frame['root_text'].map(
    lambda x: string_to_list(x)
)

# substitute comments with a number representing # of words in it
data_frame['comments'] = \
    data_frame['comments'].map(lambda x: len(x.split(' ')))

# remove duplicates from root_text lists
data_frame['root_text'] = \
    data_frame['root_text'].map(lambda x: len(list(dict.fromkeys(x))))

reviewers_stats = build_reviewer_stats(data_frame.groupby(['reviewer_id']))
listing_stats = build_listings_stats(data_frame.groupby(['listing_id']))

reviewers_stats.to_csv(reviewers_stats_path, sep=";")
listing_stats.to_csv(listings_stats_path, sep=";")
