from rvquality.component import Component

class MySimpleComponent(Component):
  def __init__(self, weight=1):
    super().__init__(weight, "MySimpleComponent")

  def apply(self, main_tbl, meta_tbl, r_means_tbl, l_means_tbl, r_diffs_tbl, l_diffs_tbl, rv_id):
    return rv_id

class TestComponent(object):
  def test_application(self):
    assert MySimpleComponent().apply(None, None, None, None, None, None, 1) == 1

  def test_weight(self):
    k_weight = 39
    assert MySimpleComponent(k_weight).weight == k_weight

  def test_name(self):
    assert MySimpleComponent().name == "MySimpleComponent"

  def test_nulls_zero(self):
    assert MySimpleComponent(0).apply(None, None, None, None, None, None, 1) == 0