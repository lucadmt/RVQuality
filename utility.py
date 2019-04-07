#!/usr/bin/env python3

'''
utility.py: Provides an utility value for each review.
'''

import pandas as pd
import threading
import csv

from preprocessing import string_to_list
from paths import nostopwords_full_path, utility_linearw_path

data_frame = pd.read_csv(nostopwords_full_path, sep=";")

utility = pd.DataFrame(data={
    "reviewer_id": data_frame['reviewer_id'],
    "listing_id": data_frame['listing_id'],
    "review_id": data_frame['id'],
    "polarity": data_frame['polarity'],
    "review_length": data_frame['comments'].map(lambda x: len(x.split(" "))),
    "review_rt_len": data_frame['root_text'].map(lambda x: len(string_to_list(x)))
})

usr_lst = pd.Series(data=data_frame['reviewer_id'].unique()).sort_values()
itm_lst = pd.Series(data=data_frame['listing_id'].unique()).sort_values()


def user_reviews(user_id):  # rev_user
    return utility.loc[utility['reviewer_id'] == user_id]


def item_reviews(item_id):  # rev_item
    return utility.loc[utility['listing_id'] == item_id]


def mean(focus_set):
    return focus_set.sum() / focus_set.count()


reviewers_means = pd.DataFrame(data={
    "reviewer_id": usr_lst,
    "polarity": usr_lst.map(lambda x: mean(user_reviews(x)['polarity'])),
    "review_length": usr_lst.map(lambda x: mean(user_reviews(x)['review_length'])),
    "review_rt_len": usr_lst.map(lambda x: mean(user_reviews(x)['review_rt_len'])),
})

items_means = pd.DataFrame(data={
    "listing_id": itm_lst,
    "polarity": itm_lst.map(lambda x: mean(item_reviews(x)['polarity'])),
    "review_length": itm_lst.map(lambda x: mean(item_reviews(x)['review_length'])),
    "review_rt_len": itm_lst.map(lambda x: mean(item_reviews(x)['review_rt_len'])),
})


def item_mean(itm, focus):
    return items_means.loc[items_means['listing_id'] == itm][focus].iloc[0]


def reviewer_mean(usr, focus):
    return reviewers_means.loc[reviewers_means['reviewer_id'] == usr][focus].iloc[0]


def reviewer_abs_diff(focus):
    return utility.apply(
        lambda row: abs(
            row[focus] - reviewer_mean(row['reviewer_id'], focus)
        ), axis=1)


def item_abs_diff(focus):
    return utility.apply(
        lambda row: abs(
            row[focus] - item_mean(row['listing_id'], focus)
        ), axis=1)


reviewers_diffs = pd.DataFrame(data={
    "review_id": utility['review_id'].sort_values(),
    "polarity": reviewer_abs_diff("polarity"),
    "review_length": reviewer_abs_diff("review_length"),
    "review_rt_len": reviewer_abs_diff("review_rt_len")
})

items_diffs = pd.DataFrame(data={
    "review_id": utility['review_id'].sort_values(),
    "polarity": item_abs_diff("polarity"),
    "review_length": item_abs_diff("review_length"),
    "review_rt_len": item_abs_diff("review_rt_len")
})


def row(review_id):
    return utility.loc[utility['review_id'] == review_id].iloc[0]


def item(review_id):
    return utility.loc[utility['review_id'] == review_id]['listing_id'].iloc[0]


def user(review_id):
    return utility.loc[utility['review_id'] == review_id]['reviewer_id'].iloc[0]


def displacement(review_id, pov, focus):
    # pov in ['item', 'user']
    # focus in ['polarity', 'review_length', 'review_rt_length']
    m_focus_pov = (
        reviewer_mean(user(review_id), focus),
        item_mean(item(review_id), focus)
    )[pov == 'item']

    review_focus_disp = abs(row(review_id)[focus] - m_focus_pov)

    max_disp = (
        reviewers_diffs[focus].max(),
        items_diffs[focus].max()
    )[pov == 'item']

    return review_focus_disp / max_disp


def review_utility(r, w):
    return sum([w[0] * displacement(r, 'user', 'polarity'),
                w[1] * displacement(r, 'item', 'polarity'),
                w[2] * displacement(r, 'user', 'review_length'),
                w[3] * displacement(r, 'item', 'review_length'),
                w[4] * displacement(r, 'user', 'review_rt_len'),
                w[5] * displacement(r, 'item', 'review_rt_len')]
               ) / sum(w)


def compute_utility(w, head_name, df):
    df[head_name] = df['id'].map(
        lambda r: review_utility(r, w))


def main():
    print("starting main()")

    t_linear = threading.Thread(target=compute_utility, args=[
        [1, 1, 1, 1, 1, 1],
        "utility_linear",
        data_frame
    ])

    t_linear_no_rt = threading.Thread(target=compute_utility, args=[
        [1, 1, 1, 1, 0, 0],
        "utility_linear_no_root_text",
        data_frame
    ])

    t_linear_no_len = threading.Thread(target=compute_utility, args=[
        [1, 1, 0, 0, 1, 1],
        "utility_linear_no_length",
        data_frame
    ])

    t_linear.start()
    t_linear_no_rt.start()
    t_linear_no_len.start()

    t_linear_no_len.join()
    t_linear_no_rt.join()
    t_linear.join()

    data_frame.to_csv(utility_linearw_path, sep=';', quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()
