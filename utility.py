#!/usr/bin/env python3

'''
utility.py: Provides an utility value for each review.
'''

import pandas as pd

from preprocessing import string_to_list
from paths import nostopwords_path

data_frame = pd.read_csv(nostopwords_path, sep=";")

utility = pd.DataFrame(data={
    "reviewer_id": data_frame['reviewer_id'],
    "listing_id": data_frame['listing_id'],
    "review_id": data_frame['id'],
    "polarity": data_frame['polarity'],
    "review_length": data_frame['comments'].map(lambda x: len(x.split(" "))),
    "review_rt_len": data_frame['root_text'].map(lambda x: len(string_to_list(x)))
})


def item(r):
    return utility[utility['review_id'] == r]['listing_id']


def user(r):
    return utility[utility['review_id'] == r]['reviewer_id']


def user_reviews(u):  # rev_user
    return utility[utility['reviewer_id'] == u]


def item_reviews(i):  # rev_item
    return utility[utility['listing_id'] == i]


def medium_polarity(pol_set):  # mp_*
    # pol_set is rev_*['polarity']
    return pol_set.sum()/pol_set.count()
