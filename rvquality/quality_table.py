import csv
import math
import os
from hashlib import md5
from statistics import mean
from string import punctuation

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from rvquality.common import drop_stopwords

# main table constants
ID_NAME = "review_id"
REVIEWER_ID_NAME = "reviewer_id"
LISTING_ID_NAME = "listing_id"
POLARITY_NAME = "polarity"
COMMENTS_NAME = "comments"
RATING_NAME = "rating"
TF_IDF_VECT_NAME = "tf_idf_vect"
TF_IDF_SUM_NAME = "tf_idf_sum"
TF_IDF_MEAN_NAME = "tf_idf_mean"
APPRECIATIONS_NAME = "appreciations"

# meta table constants
META_ID_NAME = "id"
META_REVIEWER_ID_NAME = "reviewer_id"
META_LISTING_ID_NAME = "listing_id"
META_POLARITY_NAME = "polarity"
META_LENGTH_NAME = "review_length"
META_RATING_NAME = "review_rating"


class QualityTable:
    def __init__(self, main_table, default_weight=1):
        self.main_table = main_table
        self.default_weight = default_weight
        self.weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def main_row(self, review_id):
        return (self.main_table.loc[
            self.main_table['id'] == review_id
        ].iloc[0])

    def set_reviewer_id(self, colname="reviewer_id"):
        REVIEWER_ID_NAME = colname

    def set_listing_id(self, colname="listing_id"):
        LISTING_ID_NAME = colname

    def set_review_id(self, colname="review_id"):
        REVIEWER_ID_NAME = colname

    def set_polarity(self, colname="polarity"):
        POLARITY_NAME = colname

    def set_review_comments(self, colname="comments"):
        COMMENTS_NAME = colname

    def set_review_rating(self, colname="rating"):
        RATING_NAME = colname

    def prepare(self):
        if not os.path.exists(os.path.expanduser("~/.rvcache/")):
            os.makedirs(os.path.expanduser("~/.rvcache/"))

        file_hash = md5(self.main_table.to_csv(
            index=False, sep=";", quoting=csv.QUOTE_ALL).encode()).hexdigest()

        if os.path.exists(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_meta.csv")):
            self.meta_table = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_meta.csv"))
            self.items_means = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_listing_means.csv"))
            self.reviewers_means = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_reviewers_means.csv"))
            self.items_diffs = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_listing_diffs.csv"))
            self.reviewers_diffs = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_reviewers_diffs.csv"))
        else:
            # generate intermediate tables
            self.meta_table = self._generate_meta_table()
            self.items_means = self._generate_means_table("listing")
            self.reviewers_means = self._generate_means_table("reviewer")
            self.items_diffs = self._generate_diffs_table("listing")
            self.reviewers_diffs = self._generate_diffs_table("reviewer")

            # save intermediate tables
            self.meta_table.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash+"_meta.csv"),
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.items_means.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_listing_means.csv"),
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.reviewers_means.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_reviewers_means.csv"),
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.items_diffs.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_listing_diffs.csv"),
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)
            self.reviewers_diffs.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_reviewers_diffs.csv"),
                index=False,
                sep=";", quoting=csv.QUOTE_ALL)

        if TF_IDF_VECT_NAME not in self.main_table.columns:
            self.generate_tf_idf()

    def _generate_meta_table(self):
        return pd.DataFrame(data={
            META_REVIEWER_ID_NAME: self.main_table[REVIEWER_ID_NAME],
            META_LISTING_ID_NAME: self.main_table[LISTING_ID_NAME],
            META_ID_NAME: self.main_table[ID_NAME],
            META_POLARITY_NAME: self.main_table[POLARITY_NAME],
            META_LENGTH_NAME: self.main_table[
                COMMENTS_NAME
            ].map(
                lambda x: len(x.split(" "))
            ),
            META_RATING_NAME: self.main_table[
                RATING_NAME
            ]
        })

    def meta_row(self, review_id):
        return self.meta_table.loc[
            self.meta_table[META_ID_NAME] == review_id
        ].iloc[0]

    def meta_user_reviews(self, user_id):  # rev_user
        return self.meta_table.loc[
            self.meta_table[META_REVIEWER_ID_NAME] == user_id
        ]

    def meta_item_reviews(self, item_id):  # rev_item
        return self.meta_table.loc[
            self.meta_table[META_LISTING_ID_NAME] == item_id
        ]

    def meta_item(self, review_id):
        return self.meta_table.loc[
            self.meta_table[META_ID_NAME] == review_id
        ][META_LISTING_ID_NAME].iloc[0]

    def meta_user(self, review_id):
        return self.meta_table.loc[
            self.meta_table[META_ID_NAME] == review_id
        ][META_REVIEWER_ID_NAME].iloc[0]

    def _generate_means_table(self, focus):
        focus_list = pd.Series(data=self.meta_table
                               [focus+'_id'].unique()).sort_values()
        reviews = self.meta_item_reviews \
            if focus == "listing" \
            else self.meta_user_reviews

        return pd.DataFrame(data={
            focus+"_id": focus_list,
            "polarity": focus_list.map(
                lambda x: mean((reviews(x)[META_POLARITY_NAME]).tolist())
            ),
            "review_length": focus_list.map(
                lambda x: mean(
                    (reviews(x)[META_LENGTH_NAME]).tolist())
            ),
            "review_rating": focus_list.map(
                lambda x: mean(
                    (reviews(x)[META_RATING_NAME]).tolist())
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
            "review_id": self.meta_table[
                META_ID_NAME].sort_values(),
            "polarity": self.diff("polarity"),
            "review_length": self.diff("review_length"),
            "review_rating": self.diff("review_rating")
        })

    def generate_tf_idf(self):
        vectorizer = TfidfVectorizer(
            smooth_idf=True,
            ngram_range=(1, 1),
            token_pattern=r'\S{3,}'
        )

        # split to a list of lowercased words, then drop stopwords
        cleaned_comments = self.main_table[COMMENTS_NAME].map(
            lambda x: x.lower().translate(
                str.maketrans('', '', punctuation+'\n\r\xa0\xad')
            ).split(' ')
        )

        cleaned_comments = pd.Series(
            map(drop_stopwords, cleaned_comments)
        )

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        self.main_table['lemmas'] = cleaned_comments.map(
            lambda words:
            list(map(lambda word: lemmatizer.lemmatize(word), words))
        )

        # remove empty string
        self.main_table['lemmas'] = self.main_table['lemmas'].map(
            lambda list_row:
            list(
                filter(
                    lambda word: len(word) >= 3, list_row
                )
            )
        )

        # build the corpus
        string_lemmas = self.main_table['lemmas'].map(
            lambda list_row: " ".join(list_row))
        corpus = [x for x in string_lemmas]

        sparse = vectorizer.fit_transform(corpus)

        # convert the sparse matrix to dataframe
        cols = vectorizer.get_feature_names()
        dense = sparse.todense()
        tf_idf_df = pd.DataFrame(dense, columns=cols)

        def get_tf_idf_vect(tf_idf_matrix, lemma_list, idx):
            return list(map(lambda word: tf_idf_matrix[word][idx], lemma_list))

        lemmas = self.main_table['lemmas']

        self.main_table[TF_IDF_VECT_NAME] = pd.Series(
            [
                get_tf_idf_vect(tf_idf_df, lemmas[idx], idx)
                for idx in range(0, len(self.main_table.values))
            ]
        )

        self.main_table[TF_IDF_MEAN_NAME] = \
            self.main_table[TF_IDF_VECT_NAME].map(
            lambda row: float(sum(row)) / max(len(row), 1))

        self.main_table[TF_IDF_SUM_NAME] = \
            self.main_table[TF_IDF_VECT_NAME].map(
            sum)

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
                self.meta_row(review_id)[META_RATING_NAME] -
                self.meta_row(review_id)[META_POLARITY_NAME]
            ) / 4)

    def c10(self, args):
        # args[0] = review_id
        return self._normalize_log(
            self.meta_row(args[0])[META_RATING_NAME]
        )

    def c11(self, args):
        review_id = args[0]
        tf_idf_vect = self.main_row(review_id)[TF_IDF_VECT_NAME]
        return mean(tf_idf_vect)

    def c12(self, args):
        # args[0] = review_id
        return self._normalize_log(
            self.meta_row(args[0])[META_LENGTH_NAME]
        )

    def filter_by_listing(self, listing_id):
        return self.main_table[
            self.main_table[LISTING_ID_NAME] == listing_id
        ]

    def _max_item_appreciations(self, listing_id):
        return self.filter_by_listing(listing_id)[APPRECIATIONS_NAME].max()

    def fcontr(self, review_id):
        listing_id = self.main_row(review_id)[LISTING_ID_NAME]
        if self.main_row(review_id)[APPRECIATIONS_NAME] == 0:
            return 0
        else:
            return (
                self.main_row(review_id)[APPRECIATIONS_NAME] /
                self._max_item_appreciations(listing_id)
            )

    def _null_zero_terms(self, weight, func, args):
        if weight == 0:
            return 0
        else:
            return weight * func(args)

    def review_utility(self, r, w):
        return sum([
            self._null_zero_terms(
                w[0], self._displacement, [r, 'user', 'polarity']),
            self._null_zero_terms(
                w[1], self._displacement, [r, 'item', 'polarity']),
            self._null_zero_terms(
                w[2], self._displacement, [r, 'user', 'review_length']),
            self._null_zero_terms(
                w[3], self._displacement, [r, 'item', 'review_length']),
            self._null_zero_terms(
                w[4], self._displacement, [r, 'user', 'review_rt_len']),
            self._null_zero_terms(
                w[5], self._displacement, [r, 'item', 'review_rt_len']),
            self._null_zero_terms(
                w[6], self._displacement, [r, 'user', 'review_rating']),
            self._null_zero_terms(
                w[7], self._displacement, [r, 'item', 'review_rating']),
            self._null_zero_terms(
                w[8], self.c9, [r]),
            self._null_zero_terms(
                w[9], self.c10, [r]),
            self._null_zero_terms(
                w[10], self.c11, [r]),
            self._null_zero_terms(
                w[11], self.c12, [r])]
        ) / sum(w)

    def compute_c(self, n, df, weight=1):
        local_weights = self.weights[:]  # copy template list
        local_weights[n-1] = weight
        df["c_"+n] = df['id'].map(
            lambda r: self.review_utility(r, local_weights)
        )

    def compute_utility(self, w, head_name, df):
        df[head_name] = df['id'].map(
            lambda r: self.review_utility(r, w)
        )

    def compute_fcontr(self, df):
        df['fcontr'] = df['id'].map(
            lambda r: self.fcontr(r)
        )
