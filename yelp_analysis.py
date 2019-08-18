#!/usr/bin/env python3

'''
yelp_analysis.py: Analyze the utility components on yelp dataset.
'''

import pandas as pd
import threading
import csv
import ast
import time

from rvquality.quality_table import QualityTable
from rvquality.options import Options
import rvquality.components as components
import paths

start = time.time()

# Preprocess some rows (convert back to list form)

data_frame = pd.read_csv(paths.yelp_components, sep=";")

data_frame['tf_idf_vect'] = data_frame['tf_idf_vect'].map(
    ast.literal_eval)

opts = Options()

opts.ID_NAME = "id"

# options are singleton, hence there's no need to pass it
quality_table = QualityTable(data_frame)

quality_table.prepare()

data_frame[components.C1Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C1Log()])
)
data_frame[components.C2Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C2Log()])
)
data_frame[components.C3Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C3Log()])
)
data_frame[components.C4Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C4Log()])
)
data_frame[components.C7Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C7Log()])
)
data_frame[components.C8Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C8Log()])
)
data_frame[components.C9Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C9Log()])
)
data_frame[components.C11Log().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C11Log()])
)
data_frame[components.fcontr.FcontrLog().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.fcontr.FcontrLog()])
)
data_frame[components.fcontr.FcontrGlob().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.fcontr.FcontrGlob()])
)

end = time.time()

print("computations finished, they took: " +
      str(end-start)+". Check dataset dir.")


data_frame = data_frame[
    ["id",
     "listing_id",
     "reviewer_id",
     "review.date",
     "comments",
     "pol_rating",
     "pol_rating_vader",
     "appreciations",
     "lemmas",
     "tf_idf_vect",
     "polarity",
     "est_rating",
     "tf_idf_mean",
     "rating",
     "c1",
     "C1Log",
     "c2",
     "C2Log",
     "c3",
     "C3Log",
     "c4",
     "C4Log",
     "c7",
     "C7Log",
     "c8",
     "C8Log",
     "c9",
     "C9Log",
     "c10",
     "c11",
     "C11Log",
     "c12",
     "fcontr",
     "FcontrLog",
     "FcontrGlob"]
]


data_frame.to_csv(paths.yelp_components,
                  index=False, sep=';', quoting=csv.QUOTE_ALL)
