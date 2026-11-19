import numpy as np
from src.codes import bb_code
from src.sim import sample_errors, syndromes

hx, hz = bb_code(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
rng = np.random.default_rng(1)

err = sample_errors(72, 0.05, 10000, rng)
print(err.shape, err.dtype)       
print(err.sum(axis=1).mean())     

s = syndromes(hz, err)
print(s.shape)                   

# hand-made errors:
zero = np.zeros((1, 72), dtype=np.uint8)
print(syndromes(hz, zero).sum()) 
one = np.zeros((1, 72), dtype=np.uint8); one[0, 5] = 1
print(syndromes(hz, one).sum())   