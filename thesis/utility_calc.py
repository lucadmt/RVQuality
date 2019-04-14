#!/usr/bin/env python3

'''
utility_calc.py: Provides an utility value for each review.
                Then adds all up to the main dataset
'''

import pandas as pd
import threading
import csv
import time

from common.transformations import string_to_list
from utility.utility_table import UtilityTable
import paths

start = time.time()

data_frame = pd.read_csv(paths.nostopwords_full_path, sep=";")

utility_table = UtilityTable(data_frame)

ulw = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 1, 1, 1, 1, 1, 1, 1, 1], 'utility_linearw', data_frame
])

ulw_1 = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 0, 0, 0, 0, 0, 0, 0, 0], 'utility_linearw_c1', data_frame
])

ulw_2 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 1, 0, 0, 0, 0, 0, 0, 0], 'utility_linearw_c2', data_frame
])

ulw_3 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 1, 0, 0, 0, 0, 0, 0], 'utility_linearw_c3', data_frame
])

ulw_4 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 1, 0, 0, 0, 0, 0], 'utility_linearw_c4', data_frame
])

ulw_5 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 1, 0, 0, 0, 0], 'utility_linearw_c5', data_frame
])

ulw_6 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 1, 0, 0, 0], 'utility_linearw_c6', data_frame
])

ulw_7 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 1, 0, 0], 'utility_linearw_c7', data_frame
])

ulw_8 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 1, 0], 'utility_linearw_c8', data_frame
])

ulw_9 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 0, 0, 1], 'utility_linearw_c9', data_frame
])


ulw_12 = threading.Thread(target=utility_table.compute_utility, args=[
    [1, 1, 0, 0, 0, 0, 0, 0, 0], 'utility_linearw_c12', data_frame
])

ulw_34 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 1, 1, 0, 0, 0, 0, 0], 'utility_linearw_c34', data_frame
])

ulw_56 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 1, 1, 0, 0, 0], 'utility_linearw_c56', data_frame
])

ulw_78 = threading.Thread(target=utility_table.compute_utility, args=[
    [0, 0, 0, 0, 0, 0, 1, 1, 0], 'utility_linearw_c78', data_frame
])

ulw.start()
ulw_1.start()
ulw_2.start()
ulw_3.start()
ulw_4.start()
ulw_5.start()
ulw_6.start()
ulw_7.start()
ulw_8.start()
ulw_9.start()
ulw_12.start()
ulw_34.start()
ulw_56.start()
ulw_78.start()

ulw.join()
print("ulw finished.")
ulw_1.join()
print("ulw1 finished.")
ulw_2.join()
print("ulw2 finished.")
ulw_3.join()
print("ulw3 finished.")
ulw_4.join()
print("ulw4 finished.")
ulw_5.join()
print("ulw5 finished.")
ulw_6.join()
print("ulw6 finished.")
ulw_7.join()
print("ulw7 finished.")
ulw_8.join()
print("ulw8 finished.")
ulw_9.join()
print("ulw9 finished.")
ulw_12.join()
print("ulw12 finished.")
ulw_34.join()
print("ulw34 finished.")
ulw_56.join()
print("ulw56 finished.")
ulw_78.join()
print("ulw78 finished.")

end = time.time()
print("computations finished, they took: " +
      str(end-start)+". Check dataset dir.")

data_frame.to_csv("../dataset/utility_linearw_20_full.csv",
                  index=False, sep=';', quoting=csv.QUOTE_ALL)
