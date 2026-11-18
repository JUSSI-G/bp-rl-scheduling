import numpy as np


def shift(n):
    matrix = np.eye(n, dtype=np.uint8)
    return np.roll(matrix, 1, axis=1)

