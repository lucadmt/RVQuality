#!/usr/bin/env python3

'''
yelp_analysis.py: Analyze the utility components on yelp dataset.
'''

import pandas as pd
import threading
import csv
import time

from thesis.common.transformations import string_to_list, switch_series
from thesis.utility import UtilityTable
import paths

'''
Index(['id', 'listing_id', 'reviewer_id', 'review.date', 'comments',
       'review.useful', 'review.funny', 'review.cool', 'review.stars',
       'est_rating', 'polarity', 'pol_rating', 'continuous_pol',
       'polarity_vader', 'pol_rating_vader', 'continuous_pol_vader',
       'pol_mean_tb_v', 'appreciations'],
      dtype='object')
'''

start = time.time()

data_frame = pd.read_csv(paths.yelp_path, sep=";")

yelp_components = pd.DataFrame()

# adjust dataframe to compute with textblob polarity
data_frame = switch_series(data_frame, "polarity", "continuous_pol")
utility_table = UtilityTable(data_frame)

c1tb_t = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 0, 0, 0, 0, 0, 0, 0, 0], 'c1tb', data_frame
])

c2tb_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 1, 0, 0, 0, 0, 0, 0, 0], 'c2tb', data_frame
])

c1tb_t.start()
c2tb_t.start()

c1tb_t.join()
c2tb_t.join()

# adjust dataframe to compute with vader polarity
data_frame = switch_series(data_frame, "polarity", "continuous_pol_vader")
utility_table = UtilityTable(data_frame)

c1v_t = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 0, 0, 0, 0, 0, 0, 0, 0], 'c1v', data_frame
])

c2v_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 1, 0, 0, 0, 0, 0, 0, 0], 'c2v', data_frame
])

c1v_t.start()
c2v_t.start()

c1v_t.join()
c2v_t.join()

# adjust dataframe to compute with mean polarity
data_frame = switch_series(data_frame, "polarity", "pol_mean_tb_v")
utility_table = UtilityTable(data_frame)

c1mean_t = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 0, 0, 0, 0, 0, 0, 0, 0], 'c1mean', data_frame
])

c2mean_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 1, 0, 0, 0, 0, 0, 0, 0], 'c2mean', data_frame
])

c1mean_t.start()
c2mean_t.start()

c1mean_t.join()
c2mean_t.join()

# compute the other components for yelp dataset

c3_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 1, 0, 0, 0, 0, 0, 0], 'c3', data_frame
])

c4_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 1, 0, 0, 0, 0, 0], 'c4', data_frame
])

c3_t.start()
c4_t.start()

c3_t.join()
c4_t.join()

# review_rating = review.stars

# review_rating = appreciations
data_frame = switch_series(data_frame, "est_rating", "review.stars")
utility_table = UtilityTable(data_frame)

c7stars_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 1, 0, 0], 'c7stars', data_frame
])

c8stars_t = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 'c8stars', data_frame
])

c7stars_t.start()
c8stars_t.start()

c7stars_t.join()
c8stars_t.join()

end = time.time()

print("computations finished, they took: " +
      str(end-start)+". Check dataset dir.")

yelp_components['c1tb'] = data_frame['c1tb']
yelp_components['c2tb'] = data_frame['c2tb']
yelp_components['c1v'] = data_frame['c1v']
yelp_components['c2v'] = data_frame['c2v']
yelp_components['c1mean'] = data_frame['c1mean']
yelp_components['c2mean'] = data_frame['c2mean']
yelp_components['c3'] = data_frame['c3']
yelp_components['c4'] = data_frame['c4']
yelp_components['c7stars'] = data_frame['c7stars']
yelp_components['c8stars'] = data_frame['c8stars']


yelp_components.to_csv(paths.yelp_polarities_single_path,
                       index=False, sep=';', quoting=csv.QUOTE_ALL)
