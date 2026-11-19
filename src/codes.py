import numpy as np


def shift(n):
    matrix = np.eye(n, dtype=np.uint8)
    return np.roll(matrix, 1, axis=1)

def monomial(l, m, a, b):
    x = np.kron(shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), shift(m))
    return np.linalg.matrix_power(x, a) @ np.linalg.matrix_power(y, b)