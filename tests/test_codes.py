import numpy as np
from src.codes import bb_code, gf2_rank, gf2_kernel
from src.sim import failures


def test_gf2_rank_small():
    T = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    assert gf2_rank(T) == 2


def check_bb(l, m, A, B, n, k):
    hx, hz = bb_code(l, m, A, B)
    assert hx.shape == (n // 2, n)
    assert (hx.sum(axis=1) == 6).all()
    assert (hx.sum(axis=0) == 3).all()
    assert not (hx @ hz.T % 2).any()
    assert gf2_rank(hx) == gf2_rank(hz)
    assert n - gf2_rank(hx) - gf2_rank(hz) == k


def test_bb_72():
    check_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 72, 12)


def test_bb_144():
    check_bb(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 144, 12)


def test_kernel():
    hx, hz = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    K = gf2_kernel(hx)
    assert K.shape == (72 - gf2_rank(hx), 72)
    assert not (hx @ K.T % 2).any()


def test_failures():
    hx, hz = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    K = gf2_kernel(hx)
    e = np.zeros((1, 72), dtype=np.uint8)
    e[0, 3] = 1
    assert not failures(hz, K, e, e)[0]
    assert not failures(hz, K, e, e ^ hx[0:1])[0]
    assert failures(hz, K, e, np.zeros_like(e))[0]
    logicals = gf2_kernel(hz)
    flagged = failures(hz, K, np.zeros_like(logicals), logicals)
    assert flagged.any()
