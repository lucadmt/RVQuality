import pandas as pd
from common.transformations import string_to_list


def generate_meta_table(full_table):
    return pd.DataFrame(data={
        "reviewer_id": full_table['reviewer_id'],
        "listing_id": full_table['listing_id'],
        "review_id": full_table['id'],
        "polarity": full_table['polarity'],
        "review_length": full_table['comments'].map(
            lambda x: len(x.split(" "))
        ),
        "review_rt_len": full_table['root_text'].map(
            lambda x: len(string_to_list(x))
        ),
        "review_rating": full_table['est_rating']
    })
