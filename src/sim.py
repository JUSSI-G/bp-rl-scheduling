import numpy as np


def sample_errors(n, p, shots, rng):
    r = rng.random((shots, n))
    return (r < p).astype(np.uint8)


def syndromes(hz, err):
    return err @ hz.T % 2


def failures(hz, K, err, ehat):
    resid = err ^ ehat
    not_converged = (resid @ hz.T % 2).any(axis=1)
    logical = (resid @ K.T % 2).any(axis=1)
    return not_converged | logical