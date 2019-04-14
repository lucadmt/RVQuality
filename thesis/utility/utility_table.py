import pandas as pd
from multiprocessing.pool import ThreadPool

from .mean_table import MeanTable
from .diff_table import DiffTable

from .common import generate_meta_table


class UtilityTable:
    def __init__(self, *args, **kwargs):
        self.main_table = args[0]
        self.meta_table = generate_meta_table(self.main_table)

        pool = ThreadPool(processes=4)
        rv_m = pool.apply_async(
            self.__get_table, [self.meta_table, 'reviewer', 0])
        itm_m = pool.apply_async(
            self.__get_table, [self.meta_table, 'listing', 0])
        rv_d = pool.apply_async(
            self.__get_table, [self.meta_table, 'reviewer', 1])
        itm_d = pool.apply_async(
            self.__get_table, [self.meta_table, 'listing', 1])

        self.items_means = itm_m.get().get()
        self.reviewers_means = rv_m.get().get()
        self.items_diffs = itm_d.get().get()
        self.reviewers_diffs = rv_d.get().get()

    def __get_table(self, meta, focus, type):
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

    def item(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ]['listing_id'].iloc[0]

    def user(self, review_id):
        return self.meta_table.loc[
            self.meta_table['review_id'] == review_id
        ]['reviewer_id'].iloc[0]

    def displacement(self, review_id, pov, focus):
        # pov in ['item', 'user']
        # focus in ['polarity', 'review_length', 'review_rt_length']
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

    def c9(self, review_id):
        return 1 - (
            abs(
                self.row(review_id)['review_rating'] -
                self.row(review_id)['polarity']
            ) / 4)

    def __null_zero_terms(self, weight, func):
        if weight == 0:
            return weight
        else:
            return weight * func

    def review_utility(self, r, w):
        return sum([
            self.__null_zero_terms(
                w[0], self.displacement(r, 'user', 'polarity')),
            self.__null_zero_terms(
                w[1], self.displacement(r, 'item', 'polarity')),
            self.__null_zero_terms(
                w[2], self.displacement(r, 'user', 'review_length')),
            self.__null_zero_terms(
                w[3], self.displacement(r, 'item', 'review_length')),
            self.__null_zero_terms(
                w[4], self.displacement(r, 'user', 'review_rt_len')),
            self.__null_zero_terms(
                w[5], self.displacement(r, 'item', 'review_rt_len')),
            self.__null_zero_terms(
                w[6], self.displacement(r, 'user', 'review_rating')),
            self.__null_zero_terms(
                w[7], self.displacement(r, 'item', 'review_rating')),
            self.__null_zero_terms(
                w[8], self.c9(r))]
        ) / sum(w)

    def compute_utility(self, w, head_name, df):
        df[head_name] = df['id'].map(
            lambda r: self.review_utility(r, w))
