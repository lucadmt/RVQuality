import pandas as pd
from statistics import mean
import math
import csv
import os
from hashlib import md5


class QualityTable:
    def __init__(self, main_table, default_weight=1):
        self.main_table = main_table
        self.default_weight = default_weight

    def main_row(self, review_id):
        return (self.main_table.loc[
            self.main_table['id'] == review_id
        ].iloc[0])

    def set_reviewer_id(self, colname="reviewer_id"):
        self.reviewer_id_name = colname

    def set_listing_id(self, colname="listing_id"):
        self.listing_id_name = colname

    def set_review_id(self, colname="review_id"):
        self.review_id_name = colname

    def set_polarity(self, colname="polarity"):
        self.polarity_name = colname

    def set_review_comments(self, colname="comments"):
        self.review_comments_name = colname

    def prepare(self):
        if not os.path.exists("~/.rvcache"):
            os.makedirs("~/.rvcache")
        file_hash = md5(self.main_table.to_csv(
            index=False, sep=";", quoting=csv.QUOTE_ALL))
        if os.path.exists("~/.rvcache/"+file_hash+".csv"):
            self.meta_table = pd.read_csv("~/.rvcache/"+file_hash+"_meta.csv")
            self.items_means = pd.read_csv(
                "~/.rvcache/"+file_hash+"_listing_means.csv")
            self.reviewers_means = pd.read_csv(
                "~/.rvcache/"+file_hash+"_reviewers_means.csv")
            self.items_diffs = pd.read_csv(
                "~/.rvcache/"+file_hash+"_listing_diffs.csv")
            self.reviewers_diffs = pd.read_csv(
                "~/.rvcache/"+file_hash+"_reviewers_diffs.csv")
        else:
            self.meta_table = self._generate_meta_table()
            self.items_means = self._generate_means_table("listing")
            self.reviewers_means = self._generate_means_table("reviewer")
            self.items_diffs = self._generate_diffs_table("listing")
            self.reviewers_diffs = self._generate_diffs_table("reviewer")

            self.meta_table.to_csv(
                "~/.rvcache"+file_hash+"_meta.csv",
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.items_means.to_csv(
                "~/.rvcache"+file_hash+"_listing_means.csv",
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.reviewers_means.to_csv(
                "~/.rvcache"+file_hash+"_reviewers_means.csv",
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.items_diffs.to_csv(
                "~/.rvcache"+file_hash+"_listing_diffs.csv",
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.reviewers_diffs.to_csv(
                "~/.rvcache"+file_hash+"_reviewers_diffs.csv",
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)

    def _generate_meta_table(self):
        return pd.DataFrame(data={
            "reviewer_id": self.main_table[self.reviewer_id_name],
            "listing_id": self.main_table[self.listing_id_name],
            "review_id": self.main_table[self.review_id_name],
            "polarity": self.main_table[self.polarity_name],
            "review_length": self.main_table[self.review_comments_name].map(
                lambda x: len(x.split(" "))
            ),
            "review_rating": self.main_table['rating']
        })

    def meta_row(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ].iloc[0]

    def meta_user_reviews(self, user_id):  # rev_user
        return self.meta_table.loc[self.meta_table['reviewer_id'] == user_id]

    def meta_item_reviews(self, item_id):  # rev_item
        return self.meta_table.loc[self.meta_table['listing_id'] == item_id]

    def meta_item(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ]['listing_id'].iloc[0]

    def meta_user(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ]['reviewer_id'].iloc[0]

    def _generate_means_table(self, focus):
        focus_list = pd.Series(data=self.meta_table
                               [focus+'_id'].unique()).sort_values()
        reviews = self.meta_item_reviews \
            if focus == "listing" \
            else self.meta_user_reviews

        return pd.DataFrame(data={
            focus+"_id": focus_list,
            "polarity": focus_list.map(
                lambda x: mean((reviews(x)['polarity']).tolist())
            ),
            "review_length": focus_list.map(
                lambda x: mean((reviews(x)['review_length']).tolist())
            ),
            "review_rating": focus_list.map(
                lambda x: mean((reviews(x)['review_rating']).tolist())
            )
        })

    def item_mean(self, itm, focus):
        return self.items_means.loc[
            self.items_means['listing_id'] == itm
        ][focus].iloc[0]

    def reviewer_mean(self, usr, focus):
        return self.reviewers_means.loc[
            self.reviewers_means['reviewer_id'] == usr
        ][focus].iloc[0]

    def _reviewer_abs_diff(self, focus):
        return self.meta_table.apply(
            lambda row: abs(
                row[focus] - self.reviewer_mean(row['reviewer_id'], focus)
            ), axis=1)

    def _item_abs_diff(self, focus):
        return self.meta_table.apply(
            lambda row: abs(
                row[focus] - self.item_mean(row['listing_id'], focus)
            ), axis=1)

    def _generate_diffs_table(self, focus):
        self.diff = self._reviewer_abs_diff \
            if focus == "reviewer" \
            else self._item_abs_diff

        return pd.DataFrame(data={
            "review_id": self.meta_table['review_id'].sort_values(),
            "polarity": self.diff("polarity"),
            "review_length": self.diff("review_length"),
            "review_rating": self.diff("review_rating")
        })

    def _displacement(self, args):
        review_id = args[0]
        pov = args[1]
        focus = args[2]
        # pov in ['item', 'user']
        # focus in ['polarity', 'review_length', 'review_rating']
        m_focus_pov = (
            self.reviewer_mean(self.meta_user(review_id), focus),
            self.item_mean(self.meta_item(review_id), focus)
        )[pov == 'item']

        review_focus_disp = abs(self.meta_row(review_id)[focus] - m_focus_pov)

        max_disp = (
            self.reviewers_diffs[focus].max(),
            self.items_diffs[focus].max()
        )[pov == 'item']

        return review_focus_disp / max_disp

    def _normalize_log(self, arg):
        return (
            math.log10(arg + 1) /
            (1 + math.log10(arg + 1))
        )

    # def c_*(self, review_id, weight)

    def c9(self, args):
        review_id = args[0]
        return 1 - (
            abs(
                self.meta_row(review_id)['review_rating'] -
                self.meta_row(review_id)['polarity']
            ) / 4)

    def c10(self, args):
        # args[0] = review_id
        return self._normalize_log(self.meta_row(args[0])['review_rating'])

    def c11(self, args):
        review_id = args[0]
        tf_idf_vect = self.main_row(review_id)['tf_idf_vect']
        return mean(tf_idf_vect)

    def c12(self, args):
        # args[0] = review_id
        return self._normalize_log(self.meta_row(args[0])['review_length'])

    def filter_by_listing(self, listing_id):
        return self.main_table[
            self.main_table['listing_id'] == listing_id
        ]

    def _max_item_appreciations(self, listing_id):
        return self.filter_by_listing(listing_id)['appreciations'].max()

    def fcontr(self, review_id):
        listing_id = self.main_row(review_id)['listing_id']
        if self.main_row(review_id)['appreciations'] == 0:
            return 0
        else:
            return (
                self.main_row(review_id)['appreciations'] /
                self._max_item_appreciations(listing_id)
            )

    def __null_zero_terms(self, weight, func, args):
        if weight == 0:
            return 0
        else:
            return weight * func(args)

    def review_utility(self, r, w):
        return sum([
            self.__null_zero_terms(
                w[0], self._displacement, [r, 'user', 'polarity']),
            self.__null_zero_terms(
                w[1], self._displacement, [r, 'item', 'polarity']),
            self.__null_zero_terms(
                w[2], self._displacement, [r, 'user', 'review_length']),
            self.__null_zero_terms(
                w[3], self._displacement, [r, 'item', 'review_length']),
            self.__null_zero_terms(
                w[4], self._displacement, [r, 'user', 'review_rt_len']),
            self.__null_zero_terms(
                w[5], self._displacement, [r, 'item', 'review_rt_len']),
            self.__null_zero_terms(
                w[6], self._displacement, [r, 'user', 'review_rating']),
            self.__null_zero_terms(
                w[7], self._displacement, [r, 'item', 'review_rating']),
            self.__null_zero_terms(
                w[8], self.c9, [r]),
            self.__null_zero_terms(
                w[9], self.c10, [r]),
            self.__null_zero_terms(
                w[10], self.c11, [r]),
            self.__null_zero_terms(
                w[11], self.c12, [r])]

        ) / sum(w)

    def compute_utility(self, w, head_name, df):
        df[head_name] = df['id'].map(
            lambda r: self.review_utility(r, w)
        )

    def compute_fcontr(self, df):
        df['fcontr'] = df['id'].map(
            lambda r: self.fcontr(r)
        )
