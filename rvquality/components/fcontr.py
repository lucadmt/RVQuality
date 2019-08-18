from rvquality.component import Component

class Fcontr(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "Fcontr")

  def _max_item_appreciations(self, main_tbl, listing_id):
    return main_tbl[
      main_tbl[self.opts.LISTING_ID_NAME] == listing_id
    ][self.opts.APPRECIATIONS_NAME].max()
  
  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    listing_id = self.main_row(main_tbl, rv_id)[self.opts.LISTING_ID_NAME]
    if self.main_row(main_tbl, rv_id)[self.opts.APPRECIATIONS_NAME] == 0:
      return 0
    else:
      return (
        self.main_row(main_tbl, rv_id)[self.opts.APPRECIATIONS_NAME] /
        self._max_item_appreciations(main_tbl, listing_id)
      )

class FcontrGlob(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "FcontrGlob")
  
  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    return (
      self.main_row(main_tbl, rv_id)[self.opts.APPRECIATIONS_NAME] /
      main_tbl[self.opts.APPRECIATIONS_NAME].max()
    )

class FcontrLog(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "FcontrLog")
  
  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    return self.normalize_log(self.main_row(main_tbl, rv_id)[self.opts.APPRECIATIONS_NAME])