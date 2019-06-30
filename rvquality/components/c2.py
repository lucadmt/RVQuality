from rvquality.component import Component

class C2(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C2")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    m_focus_itm = self.item_mean(l_means_tbl, self.meta_item(meta_tbl, rv_id), "polarity")
    review_polarity_disp = abs(self.meta_row(meta_tbl, rv_id)["polarity"] - m_focus_itm)
    max_disp = l_diffs_tbl["polarity"].max()
    return review_polarity_disp / max_disp

class C2Log(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C2Log")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    m_focus_itm = self.item_mean(l_means_tbl, self.meta_item(meta_tbl, rv_id), "polarity")
    review_polarity_disp = abs(self.meta_row(meta_tbl, rv_id)["polarity"] - m_focus_itm)
    return self.normalize_log(review_polarity_disp)