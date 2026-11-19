import numpy as np
from src.codes import bb_code

hx, hz = bb_code(6, 6, [(3,0), (0,1), (0,2)], [(0,3), (1,0), (2,0)])
print(hx.shape, hz.shape)
print(hx.sum(axis=1).min(), hx.sum(axis=1).max())
print(hx.sum(axis=0).min(), hx.sum(axis=0).max())
print((hx @ hz.T % 2).any())
cols = np.nonzero(hx[0])[0]
print([("L" if c < 36 else "R", divmod(int(c) % 36, 6)) for c in cols])