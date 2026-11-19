import numpy as np
from src.codes import shift, monomial

M = monomial(6, 6, 3, 0)
print(M.shape)
print(M.sum(axis=1).min(), M.sum(axis=1).max())
print(divmod(M[0].argmax(), 6))
print(divmod(monomial(6, 6, 0, 2)[0].argmax(), 6))
print(monomial(6, 6, 0, 0).sum())