import numpy as np


def min_max_scale(array: np.ndarray):
    return (array - array.min()) / (array.max() - array.min())


def negative_log_likelihood(y_predict, y):
    return - (y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)).mean()
