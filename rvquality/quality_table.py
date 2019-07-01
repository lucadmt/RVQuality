import csv
import os
from hashlib import md5
from statistics import mean
from string import punctuation

import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from rvquality.common import drop_stopwords
from rvquality.options import Options
import rvquality.components as components


class QualityTable:
  def __init__(self, main_table):
    self.opts = Options()
    self.main_table = main_table

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

    if os.path.exists(os.path.expanduser("~/.rvcache/"+file_hash+"_meta.csv")):
      self.meta_table = pd.read_csv(os.path.expanduser(
        "~/.rvcache/"+file_hash+"_meta.csv"), sep=";")
      self.items_means = pd.read_csv(os.path.expanduser(
        "~/.rvcache/"+file_hash+"_listing_means.csv"), sep=";")
      self.reviewers_means = pd.read_csv(os.path.expanduser(
        "~/.rvcache/"+file_hash+"_reviewers_means.csv"), sep=";")
      self.items_diffs = pd.read_csv(os.path.expanduser(
        "~/.rvcache/"+file_hash+"_listing_diffs.csv"), sep=";")
      self.reviewers_diffs = pd.read_csv(os.path.expanduser(
        "~/.rvcache/"+file_hash+"_reviewers_diffs.csv"), sep=";")
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

  def meta_user_reviews(self, user_id):  # rev_user
    return self.meta_table.loc[
      self.meta_table[self.opts.META_REVIEWER_ID_NAME] == user_id
    ]

  def meta_item_reviews(self, item_id):  # rev_item
    return self.meta_table.loc[
      self.meta_table[self.opts.META_LISTING_ID_NAME] == item_id
    ]

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
      lambda list_row: " ".join(list_row)
    )
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
      self.main_table[self.opts.TF_IDF_VECT_NAME].map(sum)

  def quality_of(self, review_id, comp_lst):
    numerator = 0
    values_lst = [
        component.apply(self.main_table,
                        self.meta_table,
                        self.reviewers_means,
                        self.items_means,
                        self.reviewers_diffs,
                        self.items_diffs,
                        review_id)
        for component in comp_lst
    ]
    numerator = sum([x for x in values_lst])
    weight_lst = [component.weight for component in comp_lst]
    denominator = sum([x for x in weight_lst])
    return numerator/denominator