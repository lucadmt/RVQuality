import pandas as pd


class MeanTable:
    def __init__(self, *args, **kwargs):
        self.meta_table = args[0]
        self.focus = str(kwargs['focus'])  # listing | reviewer
        self.focus_list = pd.Series(data=self.meta_table
                                    [self.focus+'_id'].unique()).sort_values()
        self.reviews = (self.user_reviews, self.item_reviews)[
            self.focus == 'listing'
        ]
        self.table = pd.DataFrame(data={
            self.focus+"_id": self.focus_list,
            "polarity": self.focus_list.map(
                lambda x: self.mean(self.reviews(x)['polarity'])
            ),
            "review_length": self.focus_list.map(
                lambda x: self.mean(self.reviews(x)['review_length'])
            ),
            "review_rt_len": self.focus_list.map(
                lambda x: self.mean(self.reviews(x)['review_rt_len'])
            ),
            "review_rating": self.focus_list.map(
                lambda x: self.mean(self.reviews(x)['review_rating'])
            )
        })

    def get(self):
        return self.table

    def mean(self, focus_set):
        return focus_set.sum() / focus_set.count()

    def user_reviews(self, user_id):  # rev_user
        return self.meta_table.loc[self.meta_table['reviewer_id'] == user_id]

    def item_reviews(self, item_id):  # rev_item
        return self.meta_table.loc[self.meta_table['listing_id'] == item_id]
