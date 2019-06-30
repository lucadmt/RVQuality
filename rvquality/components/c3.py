from rvquality.component import Component

class C3(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C3")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    m_focus_usr = self.reviewer_mean(r_means_tbl, self.meta_user(meta_tbl, rv_id), "review_length")
    review_length_disp = abs(self.meta_row(meta_tbl, rv_id)["review_length"] - m_focus_usr)
    max_disp = r_diffs_tbl["review_length"].max()
    return review_length_disp / max_disp

class C3Log(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C3Log")

  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    m_focus_usr = self.reviewer_mean(r_means_tbl, self.meta_user(meta_tbl, rv_id), "review_length")
    review_length_disp = abs(self.meta_row(meta_tbl, rv_id)["review_length"] - m_focus_usr)
    return self.normalize_log(review_length_disp)