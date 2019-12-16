import numpy as np


class T1Hist:
    def __init__(
        self, max_t1: float = 3000, min_t1: float = 200
    ):
        self.max_t1 = max_t1
        self.min_t1 = min_t1

    def get_t1_histogram(self, t1_matrix: np.ndarray, norm_m0_matrix: np.ndarray):
        t1_histogram = t1_matrix.astype(float).ravel()
        t1_weights = norm_m0_matrix.astype(float).ravel()
        self.t1_histogram, self.t1_weights = self.remove_outliers(
            t1_histogram, t1_weights
        )

    def remove_outliers(self, t1_histogram, t1_weights):
        t1_weights = t1_weights[
            (t1_histogram > (self.min_t1))
            & (t1_histogram < (self.max_t1))
        ]
        t1_histogram = t1_histogram[
            (t1_histogram > (self.min_t1))
            & (t1_histogram < (self.max_t1))
        ]
        return t1_histogram, t1_weights
