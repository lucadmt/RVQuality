#!/usr/bin/env python3

'''
airbnb_polarities.py: Provides different polarity values for airbnb dataset
'''

import pandas as pd
import threading
import csv
import time

from thesis.common.transformations import string_to_list
from thesis.utility import UtilityTable
import paths

start = time.time()

data_frame = pd.read_csv(paths.airbnb_vader, sep=";")

airbnb_polarities = pd.DataFrame()

data_frame['pol_mean_tb_v'] = (
    data_frame['continuous_pol'] + data_frame['continuous_pol_vader']) / 2

# adjust dataframe to compute with textblob polarity
data_frame = data_frame.drop("polarity", axis=1)
data_frame['polarity'] = data_frame['continuous_pol']
data_frame = data_frame.drop("continuous_pol", axis=1)

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

# adjust dataframe to compute with textblob polarity
data_frame = data_frame.drop("polarity", axis=1)
data_frame['polarity'] = data_frame['continuous_pol_vader']
data_frame = data_frame.drop("continuous_pol_vader", axis=1)

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

# adjust dataframe to compute with textblob polarity
data_frame = data_frame.drop("polarity", axis=1)
data_frame['polarity'] = data_frame['pol_mean_tb_v']
data_frame = data_frame.drop("pol_mean_tb_v", axis=1)

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

end = time.time()

print("computations finished, they took: " +
      str(end-start)+". Check dataset dir.")

airbnb_polarities['c1tb'] = data_frame['c1tb']
airbnb_polarities['c2tb'] = data_frame['c2tb']
airbnb_polarities['c1v'] = data_frame['c1v']
airbnb_polarities['c2v'] = data_frame['c2v']
airbnb_polarities['c1mean'] = data_frame['c1mean']
airbnb_polarities['c2mean'] = data_frame['c2mean']

data_frame.to_csv(paths.airbnb_polarities_path,
                  index=False, sep=';', quoting=csv.QUOTE_ALL)

airbnb_polarities.to_csv(paths.airbnb_polarities_single_path,
                         index=False, sep=';', quoting=csv.QUOTE_ALL)
