import numpy as np

def sample_errors(n, p, shots, rng):
    r = rng.random((shots, n))
    return (r < p).astype(np.uint8)

def syndromes(hz, err):
    return err @ hz.T % 2