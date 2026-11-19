import numpy as np
from src.codes import bb_code, gf2_rank

H = np.array([[1,1,0,0,0],
              [0,1,1,0,0],
              [1,0,1,0,0],
              [0,0,0,1,1]], dtype=np.uint8)
T = np.array([[0,1,1],
              [1,1,0],
              [1,0,1]], dtype=np.uint8)
print(gf2_rank(H), gf2_rank(T))

hx, hz = bb_code(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
print(gf2_rank(hx), gf2_rank(hz))
print(72 - gf2_rank(hx) - gf2_rank(hz))