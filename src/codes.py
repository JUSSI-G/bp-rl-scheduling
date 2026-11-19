import numpy as np


def shift(n):
    matrix = np.eye(n, dtype=np.uint8)
    return np.roll(matrix, 1, axis=1)

def monomial(l, m, a, b):
    x = np.kron(shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), shift(m))
    return np.linalg.matrix_power(x, a) @ np.linalg.matrix_power(y, b)

def bb_code(l, m, A_terms, B_terms):
    A = np.zeros((l * m, l * m), dtype=np.uint8)
    for (a, b) in A_terms:
        A = A + monomial(l, m, a, b)
    A = A % 2             
    B = np.zeros((l * m, l * m), dtype=np.uint8)
    for (a, b) in B_terms:
        B = B + monomial(l, m, a, b)
    B = B % 2
    hx = np.hstack([A, B])
    hz = np.hstack([B.T, A.T])  
    return hx, hz

def gf2_rank(M):
    M = M.copy() % 2
    m, n = M.shape
    rank = 0
    for col in range(n):
        pivot_row = None
        for row in range(rank, m):
            if M[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is not None:
            M[[rank, pivot_row]] = M[[pivot_row, rank]]
            for row in range(rank + 1, m):
                if M[row, col] == 1:
                    M[row] ^= M[rank]
            rank += 1
    return rank

def gf2_kernel(M):
    R = M.copy() % 2
    m, n = R.shape
    pivots = []
    rank = 0
    for col in range(n):
        pivot_row = None
        for row in range(rank, m):
            if R[row, col] == 1:
                pivot_row = row
                break
        if pivot_row is not None:
            R[[rank, pivot_row]] = R[[pivot_row, rank]]
            for row in range(m):
                if row != rank and R[row, col] == 1:
                    R[row] ^= R[rank]
            pivots.append(col)
            rank += 1
            if rank == m:
                break
    free = [c for c in range(n) if c not in pivots]
    K = np.zeros((len(free), n), dtype=np.uint8)
    for j, f in enumerate(free):
        K[j, f] = 1
        for i, p in enumerate(pivots):
            K[j, p] = R[i, f]
    return K