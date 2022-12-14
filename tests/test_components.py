import csv
import math
import os
import paths
from hashlib import md5
from statistics import mean
from string import punctuation

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from rvquality.common import drop_stopwords
from rvquality.options import Options
import rvquality.components as comps
import rvquality.quality_table as rv_qt

def measure(func):
  def measure_wrapper(*args, **kwargs):
    from time import clock
    start = clock()
    ret = func(*args, **kwargs)
    end = clock()
    print("%s function lasted: %.7f seconds", func.__name__, (end-start))
    return ret
  return measure_wrapper 

class QualityTable:
    def __init__(self, main_table, default_weight=1):
        self.opts = Options()
        self.main_table = main_table
        self.default_weight = default_weight
        self.weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def main_row(self, review_id):
        return (self.main_table.loc[
            self.main_table['id'] == review_id
        ].iloc[0])

    def set_reviewer_id(self, colname="reviewer_id"):
        self.opts.REVIEWER_ID_NAME = colname

    def set_listing_id(self, colname="listing_id"):
        self.opts.LISTING_ID_NAME = colname

    def set_review_id(self, colname="review_id"):
        self.opts.ID_NAME = colname

    def set_polarity(self, colname="polarity"):
        self.opts.POLARITY_NAME = colname

    def set_review_comments(self, colname="comments"):
        self.opts.COMMENTS_NAME = colname

    def set_review_rating(self, colname="rating"):
        self.opts.RATING_NAME = colname

    def prepare(self):
        if not os.path.exists(os.path.expanduser("~/.rvcache/")):
            os.makedirs(os.path.expanduser("~/.rvcache/"))

        file_hash = md5(self.main_table.to_csv(
            index=False, sep=";", quoting=csv.QUOTE_ALL).encode()).hexdigest()

        if os.path.exists(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_meta.csv")):
            self.meta_table = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_meta.csv"), sep=';')
            self.items_means = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_listing_means.csv"), sep=';')
            self.reviewers_means = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_reviewers_means.csv"), sep=';')
            self.items_diffs = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_listing_diffs.csv"), sep=';')
            self.reviewers_diffs = pd.read_csv(os.path.expanduser(
                "~/.rvcache/"+file_hash+"_reviewers_diffs.csv"), sep=';')
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
                sep=';', quoting=csv.QUOTE_ALL)
            self.items_means.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_listing_means.csv"),
                index=False,
                sep=';', quoting=csv.QUOTE_ALL)
            self.reviewers_means.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_reviewers_means.csv"),
                index=False,
                sep=';', quoting=csv.QUOTE_ALL)
            self.items_diffs.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_listing_diffs.csv"),
                index=False,
                sep=';', quoting=csv.QUOTE_ALL)
            self.reviewers_diffs.to_csv(
                os.path.expanduser("~/.rvcache/"+file_hash +
                                   "_reviewers_diffs.csv"),
                index=False,
                sep=';', quoting=csv.QUOTE_ALL)

        if self.opts.TF_IDF_VECT_NAME not in self.main_table.columns:
            self.generate_tf_idf()

    def _generate_meta_table(self):
        return pd.DataFrame(data={
            self.opts.META_REVIEWER_ID_NAME: self.main_table[self.opts.REVIEWER_ID_NAME],
            self.opts.META_LISTING_ID_NAME: self.main_table[self.opts.LISTING_ID_NAME],
            self.opts.META_ID_NAME: self.main_table[self.opts.ID_NAME],
            self.opts.META_POLARITY_NAME: self.main_table[self.opts.POLARITY_NAME],
            self.opts.META_LENGTH_NAME: self.main_table[
                self.opts.COMMENTS_NAME
            ].map(
                lambda x: len(x.split(" "))
            ),
            self.opts.META_RATING_NAME: self.main_table[
                self.opts.RATING_NAME
            ]
        })

    def meta_row(self, review_id):
        return self.meta_table.loc[
            self.meta_table[self.opts.META_ID_NAME] == review_id
        ].iloc[0]

    def meta_user_reviews(self, user_id):  # rev_user
        return self.meta_table.loc[
            self.meta_table[self.opts.META_REVIEWER_ID_NAME] == user_id
        ]

    def meta_item_reviews(self, item_id):  # rev_item
        return self.meta_table.loc[
            self.meta_table[self.opts.META_LISTING_ID_NAME] == item_id
        ]

    def meta_item(self, review_id):
        return self.meta_table.loc[
            self.meta_table[self.opts.META_ID_NAME] == review_id
        ][self.opts.META_LISTING_ID_NAME].iloc[0]

    def meta_user(self, review_id):
        return self.meta_table.loc[
            self.meta_table[self.opts.META_ID_NAME] == review_id
        ][self.opts.META_REVIEWER_ID_NAME].iloc[0]

    def _generate_means_table(self, focus):
        focus_list = pd.Series(data=self.meta_table
                               [focus+'_id'].unique()).sort_values()
        reviews = self.meta_item_reviews \
            if focus == "listing" \
            else self.meta_user_reviews

        return pd.DataFrame(data={
            focus+"_id": focus_list,
            "polarity": focus_list.map(
                lambda x: mean((reviews(x)[self.opts.META_POLARITY_NAME]).tolist())
            ),
            "review_length": focus_list.map(
                lambda x: mean(
                    (reviews(x)[self.opts.META_LENGTH_NAME]).tolist())
            ),
            "review_rating": focus_list.map(
                lambda x: mean(
                    (reviews(x)[self.opts.META_RATING_NAME]).tolist())
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
                self.opts.META_ID_NAME].sort_values(),
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
        cleaned_comments = self.main_table[self.opts.COMMENTS_NAME].map(
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

        self.main_table[self.opts.TF_IDF_VECT_NAME] = pd.Series(
            [
                get_tf_idf_vect(tf_idf_df, lemmas[idx], idx)
                for idx in range(0, len(self.main_table.values))
            ]
        )

        self.main_table[self.opts.TF_IDF_MEAN_NAME] = \
            self.main_table[self.opts.TF_IDF_VECT_NAME].map(
            lambda row: float(sum(row)) / max(len(row), 1))

        self.main_table[self.opts.TF_IDF_SUM_NAME] = \
            self.main_table[self.opts.TF_IDF_VECT_NAME].map(
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
                self.meta_row(review_id)[self.opts.META_RATING_NAME] -
                self.meta_row(review_id)[self.opts.META_POLARITY_NAME]
            ) / 4)

    def c10(self, args):
        # args[0] = review_id
        return self._normalize_log(
            self.meta_row(args[0])[self.opts.META_RATING_NAME]
        )

    def c11(self, args):
        review_id = args[0]
        tf_idf_vect = self.main_row(review_id)[self.opts.TF_IDF_VECT_NAME]
        return mean(tf_idf_vect)

    def c12(self, args):
        # args[0] = review_id
        return self._normalize_log(
            self.meta_row(args[0])[self.opts.META_LENGTH_NAME]
        )

    def filter_by_listing(self, listing_id):
        return self.main_table[
            self.main_table[self.opts.LISTING_ID_NAME] == listing_id
        ]

    def _max_item_appreciations(self, listing_id):
        return self.filter_by_listing(listing_id)[self.opts.APPRECIATIONS_NAME].max()

    def fcontr(self, review_id):
        listing_id = self.main_row(review_id)[self.opts.LISTING_ID_NAME]
        if self.main_row(review_id)[self.opts.APPRECIATIONS_NAME] == 0:
            return 0
        else:
            return (
                self.main_row(review_id)[self.opts.APPRECIATIONS_NAME] /
                self._max_item_appreciations(listing_id)
            )

    def _null_zero_terms(self, weight, func, args):
        if weight == 0:
            return 0
        else:
            return weight * func(args)

    @measure
    def summation(self, r):
      comp_lst = []
      numerator = 0
      comp_lst.append(comps.C1())
      comp_lst.append(comps.C2())
      comp_lst.append(comps.C3())
      comp_lst.append(comps.C4())
      comp_lst.append(comps.C7())
      comp_lst.append(comps.C8())
      comp_lst.append(comps.C9())
      comp_lst.append(comps.C10())
      comp_lst.append(comps.C11())
      comp_lst.append(comps.C12())
      values_lst = [component.apply(self.main_table, self.meta_table, self.reviewers_means, self.items_means, self.reviewers_diffs, self.items_diffs, r) for component in comp_lst]
      numerator = sum([x for x in values_lst])
      weight_lst = [component.weight for component in comp_lst]
      denominator = sum([x for x in weight_lst])
      return numerator/denominator

    @measure
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

class TestComponents(object):
  def test_same_values(self):
    main_table = pd.read_csv(paths.yelp_path, sep=";")
    qt = QualityTable(main_table)
    opts = Options()
    opts.POLARITY_NAME = "pol_mean_tb_v"
    opts.ID_NAME = "id"
    opts.RATING_NAME = "review.stars"
    qt.prepare()

    rv_id = main_table[qt.opts.ID_NAME][0]
    assert qt.review_utility(rv_id, [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]) == qt.summation(rv_id)
  
  def test_quality_of(self):
    main_table = pd.read_csv(paths.yelp_path, sep=";")
    qt = rv_qt.QualityTable(main_table)
    qt_o = QualityTable(main_table)
    opts = Options()
    opts.POLARITY_NAME = "pol_mean_tb_v"
    opts.ID_NAME = "id"
    opts.RATING_NAME = "review.stars"
    qt.prepare()
    qt_o.prepare()

    components = [
      comps.C1(), 
      comps.C2(), 
      comps.C3(), 
      comps.C4(), 
      comps.C7(), 
      comps.C8(),
      comps.C9(),
      comps.C10(),
      comps.C11(),
      comps.C12()]

    rv_id = main_table[qt.opts.ID_NAME][0]
    assert qt_o.summation(rv_id) == qt.quality_of(rv_id, components)