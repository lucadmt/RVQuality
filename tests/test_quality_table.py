from rvquality.quality_table import QualityTable
import os
import pandas as pd
import paths

main_table = pd.read_csv(paths.yelp_path, sep=';')
qt = QualityTable(main_table)


class TestQualityTable(object):

    def test_preparation(self):
        qt.set_polarity("pol_mean_tb_v")
        qt.set_review_id("id")
        qt.set_review_rating("review.stars")

        qt.prepare()
        assert os.path.exists("~/.rvcache")
