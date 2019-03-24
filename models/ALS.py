import numpy as np
import random
from implicit.als import AlternatingLeastSquares
from .utils import weight_cf_matrix


class ALSRecommender(AlternatingLeastSquares):

    def __init__(self,
                 factors=16,
                 dtype=np.float32,
                 iterations=2,
                 calculate_training_loss=True,
                 cf_weighting_alpha=1):
        self.cf_weighting_alpha = cf_weighting_alpha
        super().__init__(
            factors=factors,
            dtype=dtype,
            iterations=iterations,
            calculate_training_loss=calculate_training_loss)

    def fit(self, train_csr):
        #don't want to modify original incase it gets put into other models
        weighted_train_csr = weight_cf_matrix(train_csr,
                                              self.cf_weighting_alpha)
        super().fit(weighted_train_csr)