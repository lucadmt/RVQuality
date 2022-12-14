from types import FunctionType
import concurrent.futures
from rvquality.options import Options
from math import log10

class MetaComponent(type):
  def __new__(cls, name, bases, clsdict):
    if 'apply' in clsdict and type(clsdict['apply'] == FunctionType):
      def null_zero(function):
        def wrapper_null_zero(*args, **kwargs):
          if args[0].weight == 0:
            return 0
          else:
            # max_workers = None -> #processors * 5
            with concurrent.futures.ThreadPoolExecutor(max_workers = None) as executor:
              c_value = executor.submit(function, *args, **kwargs)
            return c_value.result()
        return wrapper_null_zero
      clsdict['apply'] = null_zero(clsdict['apply'])
    return type.__new__(cls, name, bases, clsdict)

class Component(object, metaclass=MetaComponent):
  def __init__(self, weight=1, name="generic component"):
    self.name = name
    self.weight = weight
    self.opts = Options()

  # selects a row in main_table by review_id
  def main_row(self, main_table, review_id):
    return (main_table.loc[
        main_table[self.opts.ID_NAME] == review_id
    ].iloc[0])

  # selects a row in mean_table by itm (listing id) in focus
  def item_mean(self, mean_tbl, itm, focus):
    return mean_tbl.loc[
          mean_tbl['listing_id'] == itm
    ][focus].iloc[0]

  # selects a row in mean_table by reviewer (reviewer id) in focus
  def reviewer_mean(self, mean_tbl, usr, focus):
      return mean_tbl.loc[
          mean_tbl['reviewer_id'] == usr
      ][focus].iloc[0]

  # selects a row from meta_table by review_id
  def meta_row(self, meta_table, review_id):
      return meta_table.loc[
          meta_table[self.opts.META_ID_NAME] == review_id
      ].iloc[0]

  # selects reviews from meta_table by reviewer (usr_id)
  def meta_user_reviews(self, meta_table, user_id):  # rev_user
      return meta_table.loc[
          meta_table[self.opts.META_REVIEWER_ID_NAME] == user_id
      ]

  # selects reviews from meta_table by listing (item_id)
  def meta_item_reviews(self, meta_table, item_id):  # rev_item
      return meta_table.loc[
          meta_table[self.opts.META_LISTING_ID_NAME] == item_id
      ]

  # selects a listing id by review_id
  def meta_item(self, meta_table, review_id):
      return meta_table.loc[
          meta_table[self.opts.META_ID_NAME] == review_id
      ][self.opts.META_LISTING_ID_NAME].iloc[0]

  # selects a reviewer id by review_id
  def meta_user(self, meta_table, review_id):
      return meta_table.loc[
          meta_table[self.opts.META_ID_NAME] == review_id
      ][self.opts.META_REVIEWER_ID_NAME].iloc[0]
  
  def normalize_log(self, arg):
    return (
        log10(arg + 1) /
        (1 + log10(arg + 1))
    )

  def apply(self, main_table, meta_table, reviewers_means, items_means, reviewers_diffs, items_diffs, review_id):
    raise NotImplementedError