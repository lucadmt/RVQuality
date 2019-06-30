from rvquality.component import Component

class C9(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C9")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    return 1 - (
      abs(
          self.meta_row(meta_tbl, rv_id)[self.opts.META_RATING_NAME] -
          self.meta_row(meta_tbl, rv_id)[self.opts.META_POLARITY_NAME]
      ) / 4
    )

class C9Log(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C9Log")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    return 1 - (
      abs(
          self.meta_row(rv_id)[self.opts.META_RATING_NAME] -
          self.meta_row(rv_id)[self.opts.META_POLARITY_NAME]
      ) / 4
    )