from rvquality.quality_table import QualityTable
import os
import pandas as pd
import paths
from rvquality.options import Options

main_table = pd.read_csv(paths.yelp_path, sep=';')
options = Options()
options.POLARITY_NAME = "pol_mean_tb_v"
options.ID_NAME = "id"
options.RATING_NAME = "review.stars"
qt = QualityTable(main_table)

class TestQualityTable(object):

    def test_preparation(self):
        qt.prepare()
        assert os.path.exists(os.path.expanduser("~/.rvcache"))

    def test_options_correct(self):
        o = Options()
        assert o.RATING_NAME == "review.stars" and o.POLARITY_NAME == "pol_mean_tb_v" and o.ID_NAME == "id"