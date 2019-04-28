import pandas as pd

import os
import sys
from thesis.common.transformations import drop_stopwords, string_to_list

script, inputfile, outputfile = sys.argv

if not os.path.isfile(inputfile):
    sys.exit("Error: No such file: "+inputfile)

data_frame = pd.read_csv(inputfile, sep=';')

if 'comments' not in data_frame.columns:
    sys.exit("Error: the file you provided hasn't 'comments' column")

data_frame['comments'] = data_frame['comments'].map(lambda x: x.split(' '))
data_frame['comments'] = drop_stopwords(data_frame['comments'])
