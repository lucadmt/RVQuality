import threading

class Options(object):
    __instance = None
    __options = {}
    __lock = threading.Lock()
    def __new__(cls):
        if Options.__instance is None:
            Options.__instance = object.__new__(cls)
        return Options.__instance

    def __init__(self):
        # main table constants
        Options.__instance.ID_NAME = "review_id"
        Options.__instance.REVIEWER_ID_NAME = "reviewer_id"
        Options.__instance.LISTING_ID_NAME = "listing_id"
        Options.__instance.POLARITY_NAME = "polarity"
        Options.__instance.COMMENTS_NAME = "comments"
        Options.__instance.RATING_NAME = "rating"
        Options.__instance.TF_IDF_VECT_NAME = "tf_idf_vect"
        Options.__instance.TF_IDF_SUM_NAME = "tf_idf_sum"
        Options.__instance.TF_IDF_MEAN_NAME = "tf_idf_mean"
        Options.__instance.APPRECIATIONS_NAME = "appreciations"

        # meta table constants
        Options.__instance.META_ID_NAME = "id"
        Options.__instance.META_REVIEWER_ID_NAME = "reviewer_id"
        Options.__instance.META_LISTING_ID_NAME = "listing_id"
        Options.__instance.META_POLARITY_NAME = "polarity"
        Options.__instance.META_LENGTH_NAME = "review_length"
        Options.__instance.META_RATING_NAME = "review_rating"