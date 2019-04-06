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


def medium(set):
    return set.sum() / set.count()


def c1(r):
    mp_user_r = medium(user_reviews(user(r))['polarity'])
    review_pol_disp = abs(r['polarity'] - mp_user_r)
    max_disp = max(
        utility.apply(
            lambda row: abs(
                row['polarity'] - medium(
                    user_reviews(row['reviewer_id'])['polarity'])
            ), axis=1))

    return review_pol_disp / max_disp


def c2(r):
    mp_item_r = medium(item_reviews(item(r))['polarity'])
    review_pol_disp = abs(r['polarity'] - mp_item_r)
    max_disp = max(
        utility.apply(
            lambda row: abs(
                row['polarity'] - medium(
                    item_reviews(row['listing_id'])['polarity'])
            ), axis=1))

    return review_pol_disp / max_disp


def c3(r):
    mlw_user_r = medium(user_reviews(user(r))['review_length'])
    review_len_disp = abs(r['review_length'] - mlw_user_r)
    max_disp = max(
        utility.apply(
            lambda row: abs(
                row['review_length'] - medium(
                    user_reviews(row['reviewer_id'])['review_length'])
            ), axis=1))

    return review_len_disp / max_disp
