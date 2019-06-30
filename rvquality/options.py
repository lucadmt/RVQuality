class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
      if cls not in cls._instances:
        cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
      return cls._instances[cls]

class Options(object, metaclass=Singleton):
  def __init__(self):
    self.ID_NAME = 'review_id'
    self.REVIEWER_ID_NAME = 'reviewer_id'
    self.LISTING_ID_NAME = 'listing_id'
    self.POLARITY_NAME = 'polarity'
    self.COMMENTS_NAME = 'comments'
    self.RATING_NAME = 'rating'
    self.TF_IDF_VECT_NAME = 'tf_idf_vect'
    self.TF_IDF_SUM_NAME = 'tf_idf_sum'
    self.TF_IDF_MEAN_NAME = 'tf_idf_mean'
    self.APPRECIATIONS_NAME = 'appreciations'
    self.META_ID_NAME = 'id'
    self.META_REVIEWER_ID_NAME = 'reviewer_id'
    self.META_LISTING_ID_NAME = 'listing_id'
    self.META_POLARITY_NAME = 'polarity'
    self.META_LENGTH_NAME = 'review_length'
    self.META_RATING_NAME = 'review_rating'