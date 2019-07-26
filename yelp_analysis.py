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

data_frame = pd.read_csv(paths.yelp_tf_idf_vect_path, sep=";")

data_frame['tf_idf_vect'] = data_frame['tf_idf_vect'].map(
    ast.literal_eval)

opts = Options()

opts.RATING_NAME = "review.stars"
opts.POLARITY_NAME = "pol_mean_tb_v"
opts.ID_NAME = "id"

# options are singleton, hence there's no need to pass it
quality_table = QualityTable(data_frame)

quality_table.prepare()

data_frame[components.C1().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C1()])
)
data_frame[components.C2().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C2()])
)
data_frame[components.C3().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C3()])
)
data_frame[components.C4().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C4()])
)
data_frame[components.C7().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C7()])
)
data_frame[components.C8().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C8()])
)
data_frame[components.C9().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C9()])
)
data_frame[components.C10().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C10()])
)
data_frame[components.C11().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C11()])
)
data_frame[components.C12().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.C12()])
)
data_frame[components.fcontr.Fcontr().name] = data_frame[opts.ID_NAME].map(
    lambda id: quality_table.quality_of(id, [components.fcontr.Fcontr()])
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
     "C1",
     "C2",
     "C3",
     "C4",
     "C7",
     "C8",
     "C9",
     "C10",
     "C11",
     "C12",
     "Fcontr"]
]


data_frame.to_csv(paths.yelp_components,
                  index=False, sep=';', quoting=csv.QUOTE_ALL)
