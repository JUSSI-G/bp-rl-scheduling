import numpy as np
from src.codes import bb_code, gf2_rank


def test_gf2_rank_small():
    T = np.array([[0,1,1],
                  [1,1,0],
                  [1,0,1]], dtype=np.uint8)
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
    check_bb(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)], 72, 12)


def test_bb_144():
    check_bb(12, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)], 144, 12)