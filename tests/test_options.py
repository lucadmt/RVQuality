import pytest
from rvquality.options import Options

class TestOptions(object):
  def test_is_singleton(self):
    opts_1 = Options()
    opts_1.dummy = 1
    opts_2 = Options()
    opts_2.dummy = 13
    assert opts_1.dummy is opts_2.dummy

  def test_get_unexistent(self):
    opts = Options()
    with pytest.raises(AttributeError) as e_info:
      opts.NULL_OPT

  def test_get_options(self):
    opts = Options()
    opts.MY = 1
    assert opts.MY == 1

  def test_preexistent(self):
    opts = Options()
    assert opts.ID_NAME == "review_id"

  def test_override_preexistent(self):
    opts = Options()
    opts.ID_NAME = "changed_id"
    assert opts.ID_NAME == "changed_id"