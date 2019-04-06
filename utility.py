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


def row(review_id):
    return utility.loc[utility['review_id'] == review_id]


def item(review_id):
    return utility.loc[utility['review_id'] == review_id]['listing_id'][0]


def user(review_id):
    return utility.loc[utility['review_id'] == review_id]['reviewer_id'][0]


def user_reviews(user_id):  # rev_user
    return utility.loc[utility['reviewer_id'] == user_id]


def item_reviews(item_id):  # rev_item
    return utility.loc[utility['listing_id'] == item_id]


def mean(focus_set):
    return focus_set.sum() / focus_set.count()


def displacement(review_id, pov, focus):
    # pov in ['item', 'user']
    # focus in ['polarity', 'review_length', 'review_rt_length']
    m_focus_pov = (mean(user_reviews(user(review_id))[focus]),
                   mean(item_reviews(item(review_id))[focus]))[pov == 'item']
    review_focus_disp = abs(row(review_id)[focus] - m_focus_pov)

    max_disp = (
        max(
            utility.apply(
                lambda row: abs(
                    row[focus] - mean(
                        user_reviews(row['reviewer_id'])[focus])
                ), axis=1)),
        max(
            utility.apply(
                lambda row: abs(
                    row[focus] - mean(
                        item_reviews(row['listing_id'])[focus])
                ), axis=1))
    )[pov == 'item']

    return review_focus_disp / max_disp


def review_utility(r, w):
    return sum([w[0] * displacement(r, 'user', 'polarity'),
                w[1] * displacement(r, 'item', 'polarity'),
                w[2] * displacement(r, 'user', 'review_length'),
                w[3] * displacement(r, 'item', 'review_length'),
                w[4] * displacement(r, 'user', 'review_rt_length'),
                w[5] * displacement(r, 'item', 'review_rt_length')]
               ) / sum(w)
