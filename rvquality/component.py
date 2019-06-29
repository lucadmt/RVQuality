from types import FunctionType
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
            return function(*args, **kwargs)
        return wrapper_null_zero
      clsdict['apply'] = null_zero(clsdict['apply'])
    return type.__new__(cls, name, bases, clsdict)

class Component(object, metaclass=MetaComponent):
  def __init__(self, weight=1, name="generic component"):
    self.name = name
    self.weight = weight
    self.opts = Options()

  def item_mean(self, mean_tbl, itm, focus):
    return mean_tbl.loc[
          mean_tbl['listing_id'] == itm
    ][focus].iloc[0]

  def reviewer_mean(self, mean_tbl, usr, focus):
      return mean_tbl.loc[
          mean_tbl['reviewer_id'] == usr
      ][focus].iloc[0]

  def meta_row(self, meta_table, review_id):
      return meta_table.loc[
          meta_table[self.opts.META_ID_NAME] == review_id
      ].iloc[0]

  def meta_user_reviews(self, meta_table, user_id):  # rev_user
      return meta_table.loc[
          meta_table[self.opts.META_REVIEWER_ID_NAME] == user_id
      ]

  def meta_item_reviews(self, meta_table, item_id):  # rev_item
      return meta_table.loc[
          meta_table[self.opts.META_LISTING_ID_NAME] == item_id
      ]

  def meta_item(self, meta_table, review_id):
      return meta_table.loc[
          meta_table[self.opts.META_ID_NAME] == review_id
      ][self.opts.META_LISTING_ID_NAME].iloc[0]

  def meta_user(self, meta_table, review_id):
      return meta_table.loc[
          meta_table[self.opts.META_ID_NAME] == review_id
      ][self.opts.META_REVIEWER_ID_NAME].iloc[0]
  
  def _normalize_log(self, arg):
    return (
        log10(arg + 1) /
        (1 + log10(arg + 1))
    )

  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    raise NotImplementedError