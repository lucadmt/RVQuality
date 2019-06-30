from rvquality.component import Component
from statistics import mean

class C11(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C11")

  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    tf_idf_vect = self.main_row(main_tbl, rv_id)[self.opts.TF_IDF_VECT_NAME]
    return mean(tf_idf_vect)

class C11Log(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "C11Log")

  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    tf_idf_vect = self.main_row(main_tbl, rv_id)[self.opts.TF_IDF_VECT_NAME]
    return self.normalize_log(mean(tf_idf_vect))