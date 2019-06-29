from rvquality.component import Component

def C2(Component):
  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    # item, polarity
    m_focus_itm = self.item_mean(l_means_tbl, self.meta_item(meta_tbl, rv_id), "polarity")
    review_polarity_disp = abs(self.meta_row(meta_tbl, rv_id)["polarity"] - m_focus_itm)
    max_disp = l_diffs_tbl["polarity"].max()
    return review_polarity_disp / max_disp