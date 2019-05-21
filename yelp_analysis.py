#!/usr/bin/env python3

'''
yelp_analysis.py: Analyze the utility components on yelp dataset.
'''

import pandas as pd
import threading
import csv
import ast
import time

from thesis.common.transformations import switch_series, drop_series
from thesis.utility import UtilityTable
import paths

'''
Index(['id', 'listing_id', 'reviewer_id', 'review.date', 'comments',
       'pol_rating', 'pol_rating_vader', 'appreciations', 'lemmas',
       'tf_idf_vect', 'polarity', 'review_rating'],
      dtype='object')
'''

start = time.time()

data_frame = pd.read_csv(paths.yelp_tf_idf_vect_path, sep=";")

data_frame['tf_idf_vect'] = data_frame['tf_idf_vect'].map(
    ast.literal_eval)

utility_table = UtilityTable(data_frame)

c1_t = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'c1', data_frame
])

c2_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'c2', data_frame
])

c1_t.start()
c2_t.start()

# compute the other components for yelp dataset

c3_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'c3', data_frame
])

c4_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'c4', data_frame
])

c3_t.start()
c4_t.start()

c7_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'c7', data_frame
])

c8_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'c8', data_frame
])

c7_t.start()
c8_t.start()

c9_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'c9', data_frame
])

c10_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'c10', data_frame
])

c11_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'c11', data_frame
])

c9_t.start()
c10_t.start()
c11_t.start()

c1_t.join()
c2_t.join()
c3_t.join()
c4_t.join()
c7_t.join()
c8_t.join()
c9_t.join()
c10_t.join()
c11_t.join()

end = time.time()

print("computations finished, they took: " +
      str(end-start)+". Check dataset dir.")


data_frame = data_frame[
    ["id", "listing_id", "reviewer_id", "review.date", "comments", "pol_rating", "pol_rating_vader", "appreciations", "lemmas",
     "tf_idf_vect", "polarity", "est_rating", "tf_idf_mean", "review.stars", "c1", "c2", "c3", "c4", "c7", "c8", "c9", "c10", "c11", ]
]


data_frame.to_csv(paths.yelp_components,
                  index=False, sep=';', quoting=csv.QUOTE_ALL)
