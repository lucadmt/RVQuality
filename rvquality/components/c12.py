from rvquality.component import Component
from statistics import mean

class C12(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C12")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    return self.normalize_log(
      self.meta_row(meta_tbl, rv_id)[self.opts.META_LENGTH_NAME]
    )