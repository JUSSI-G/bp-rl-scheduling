import numpy as np
import pytest

from src.codes import BB_CODES, bb_code, build_code, gf2_kernel, gf2_rank
from src.sim import failures


def test_gf2_rank_small():
    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    assert gf2_rank(matrix) == 2


@pytest.mark.parametrize("name, number_of_qubits", [("72", 72), ("144", 144), ("288", 288)])
def test_bb_code_parameters(name, number_of_qubits):
    l, m, a_terms, b_terms = BB_CODES[name]
    hx, hz = bb_code(l, m, a_terms, b_terms)
    assert hx.shape == (number_of_qubits // 2, number_of_qubits)
    assert (hx.sum(axis=1) == 6).all()
    assert (hx.sum(axis=0) == 3).all()
    assert not (hx @ hz.T % 2).any()
    assert gf2_rank(hx) == gf2_rank(hz)
    assert number_of_qubits - gf2_rank(hx) - gf2_rank(hz) == 12


def test_kernel():
    hx, hz, logical_test = build_code("72")
    assert logical_test.shape == (72 - gf2_rank(hx), 72)
    assert not (hx @ logical_test.T % 2).any()


def test_failures():
    hx, hz, logical_test = build_code("72")
    error = np.zeros((1, 72), dtype=np.uint8)
    error[0, 3] = 1
    assert not failures(hz, logical_test, error, error)[0]
    assert not failures(hz, logical_test, error, error ^ hx[0:1])[0]
    assert failures(hz, logical_test, error, np.zeros_like(error))[0]
    logicals = gf2_kernel(hz)
    flagged = failures(hz, logical_test, np.zeros_like(logicals), logicals)
    assert flagged.any()
