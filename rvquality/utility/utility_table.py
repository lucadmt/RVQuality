import pandas as pd
import statistics
import math
from multiprocessing.pool import ThreadPool

from .mean_table import MeanTable
from .diff_table import DiffTable

from .common import generate_meta_table


class UtilityTable:
    def __init__(self, *args, **kwargs):
        self.main_table = args[0]
        self.meta_table = generate_meta_table(self.main_table)

        # generate mean and diff tables
        pool = ThreadPool(processes=4)
        rv_m = pool.apply_async(
            self._get_table, [self.meta_table, 'reviewer', 0])
        itm_m = pool.apply_async(
            self._get_table, [self.meta_table, 'listing', 0])
        rv_d = pool.apply_async(
            self._get_table, [self.meta_table, 'reviewer', 1])
        itm_d = pool.apply_async(
            self._get_table, [self.meta_table, 'listing', 1])

        self.items_means = itm_m.get().get()
        self.reviewers_means = rv_m.get().get()
        self.items_diffs = itm_d.get().get()
        self.reviewers_diffs = rv_d.get().get()

    def _get_table(self, meta, focus, type):
        if type == 0:
            return MeanTable(meta, focus=focus)
        elif type == 1:
            return DiffTable(meta, focus=focus)
        else:
            return None

    def item_mean(self, itm, focus):
        return self.items_means.loc[
            self.items_means['listing_id'] == itm
        ][focus].iloc[0]

    def reviewer_mean(self, usr, focus):
        return self.reviewers_means.loc[
            self.reviewers_means['reviewer_id'] == usr
        ][focus].iloc[0]

    def row(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ].iloc[0]

    def main_row(self, review_id):
        return (self.main_table.loc[
            self.main_table['id'] == review_id
        ].iloc[0])

    def item(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ]['listing_id'].iloc[0]

    def user(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ]['reviewer_id'].iloc[0]

    def _normalize_log(self, arg):
        return (
            math.log10(arg + 1) /
            (1 + math.log10(arg + 1))
        )

    def c9(self, args):
        review_id = args[0]
        return 1 - (
            abs(
                self.row(review_id)['review_rating'] -
                self.row(review_id)['polarity']
            ) / 4)

    def c10(self, args):
        # args[0] = review_id
        return self._normalize_log(self.row(args[0])['review_rating'])

    def c11(self, args):
        review_id = args[0]
        tf_idf_vect = self.main_row(review_id)['tf_idf_vect']
        return statistics.mean(tf_idf_vect)

    def c12(self, args):
        # args[0] = review_id
        return self._normalize_log(self.row(args[0])['review_length'])

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

    def displacement(self, args):

        review_id = args[0]
        pov = args[1]
        focus = args[2]
        # pov in ['item', 'user']
        # focus in ['polarity', 'review_length', 'review_rating']
        m_focus_pov = (
            self.reviewer_mean(self.user(review_id), focus),
            self.item_mean(self.item(review_id), focus)
        )[pov == 'item']

        review_focus_disp = abs(self.row(review_id)[focus] - m_focus_pov)

        max_disp = (
            self.reviewers_diffs[focus].max(),
            self.items_diffs[focus].max()
        )[pov == 'item']

        return review_focus_disp / max_disp

    def __null_zero_terms(self, weight, func, args):
        if weight == 0:
            return 0
        else:
            return weight * func(args)

    def review_utility(self, r, w):
        return sum([
            self.__null_zero_terms(
                w[0], self.displacement, [r, 'user', 'polarity']),
            self.__null_zero_terms(
                w[1], self.displacement, [r, 'item', 'polarity']),
            self.__null_zero_terms(
                w[2], self.displacement, [r, 'user', 'review_length']),
            self.__null_zero_terms(
                w[3], self.displacement, [r, 'item', 'review_length']),
            self.__null_zero_terms(
                w[4], self.displacement, [r, 'user', 'review_rt_len']),
            self.__null_zero_terms(
                w[5], self.displacement, [r, 'item', 'review_rt_len']),
            self.__null_zero_terms(
                w[6], self.displacement, [r, 'user', 'review_rating']),
            self.__null_zero_terms(
                w[7], self.displacement, [r, 'item', 'review_rating']),
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
