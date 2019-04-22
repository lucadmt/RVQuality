import pandas as pd

from .mean_table import MeanTable


class DiffTable:
    def __init__(self, *args, **kwargs):
        self.meta_table = args[0]
        self.focus = str(kwargs['focus'])  # listing | reviewer
        self.focus_list = pd.Series(data=self.meta_table
                                    [self.focus+'_id'].unique()).sort_values()
        self.diff = (self.reviewer_abs_diff, self.item_abs_diff)[
            self.focus == 'listing'
        ]
        self.items_means = MeanTable(self.meta_table, focus='listing')
        self.reviewers_means = MeanTable(self.meta_table, focus='reviewer')
        self.table = pd.DataFrame(data={
            "review_id": self.meta_table['review_id'].sort_values(),
            "polarity": self.diff("polarity"),
            "review_length": self.diff("review_length"),
            "review_rating": self.diff("review_rating")
        })

    def get(self):
        return self.table

    def item_mean(self, itm, focus):
        return self.items_means.get().loc[
            self.items_means.get()['listing_id'] == itm
        ][focus].iloc[0]

    def reviewer_mean(self, usr, focus):
        return self.reviewers_means.get().loc[
            self.reviewers_means.get()['reviewer_id'] == usr
        ][focus].iloc[0]

    def reviewer_abs_diff(self, focus):
        return self.meta_table.apply(
            lambda row: abs(
                row[focus] - self.reviewer_mean(row['reviewer_id'], focus)
            ), axis=1)

    def item_abs_diff(self, focus):
        return self.meta_table.apply(
            lambda row: abs(
                row[focus] - self.item_mean(row['listing_id'], focus)
            ), axis=1)
