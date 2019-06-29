from rvquality.component import Component

def C8(Component):
  def apply(self, imain_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    # item, review_rating
    m_focus_itm = self.item_mean(l_means_tbl, self.meta_item(meta_tbl, rv_id), "review_rating")
    review_rating_disp = abs(self.meta_row(meta_tbl, rv_id)["review_rating"] - m_focus_itm)
    max_disp = l_diffs_tbl["review_rating"].max()
    return review_rating_disp / max_disp