from rvquality.component import Component

def C1(Component):
  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    # user, polarity
    m_focus_usr = self.reviewer_mean(r_means_tbl, self.meta_user(meta_tbl, rv_id), "polarity")
    review_polarity_disp = abs(self.meta_row(meta_tbl, rv_id)["polarity"] - m_focus_usr)
    max_disp = r_diffs_tbl["polarity"].max()
    return review_polarity_disp / max_disp