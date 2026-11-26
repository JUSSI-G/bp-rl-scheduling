import numpy as np

BB_CODES = {
    "72": (6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    "144": (12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    "288": (12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)]),
}


def shift(size):
    """Cyclic shift matrix of the given size."""
    identity = np.eye(size, dtype=np.uint8)
    return np.roll(identity, 1, axis=1)


def monomial(l, m, x_power, y_power):
    """Matrix of the monomial x^a y^b on the l-by-m torus."""
    x = np.kron(shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), shift(m))
    return np.linalg.matrix_power(x, x_power) @ np.linalg.matrix_power(y, y_power)


def bb_code(l, m, a_terms, b_terms):
    """Parity-check matrices hx, hz of the bivariate bicycle code with polynomials A and B."""
    a_matrix = np.zeros((l * m, l * m), dtype=np.uint8)
    for x_power, y_power in a_terms:
        a_matrix = a_matrix + monomial(l, m, x_power, y_power)
    a_matrix = a_matrix % 2
    b_matrix = np.zeros((l * m, l * m), dtype=np.uint8)
    for x_power, y_power in b_terms:
        b_matrix = b_matrix + monomial(l, m, x_power, y_power)
    b_matrix = b_matrix % 2
    hx = np.hstack([a_matrix, b_matrix])
    hz = np.hstack([b_matrix.T, a_matrix.T])
    return hx, hz


def gf2_rank(matrix):
    """Rank of a binary matrix over GF(2)."""
    reduced = matrix.copy() % 2
    number_of_rows, number_of_columns = reduced.shape
    rank = 0
    for column in range(number_of_columns):
        pivot_row = None
        for row in range(rank, number_of_rows):
            if reduced[row, column] == 1:
                pivot_row = row
                break
        if pivot_row is not None:
            reduced[[rank, pivot_row]] = reduced[[pivot_row, rank]]
            for row in range(rank + 1, number_of_rows):
                if reduced[row, column] == 1:
                    reduced[row] ^= reduced[rank]
            rank += 1
    return rank


def gf2_kernel(matrix):
    """Basis of the GF(2) null space of a binary matrix, one vector per row."""
    reduced = matrix.copy() % 2
    number_of_rows, number_of_columns = reduced.shape
    pivots = []
    rank = 0
    for column in range(number_of_columns):
        pivot_row = None
        for row in range(rank, number_of_rows):
            if reduced[row, column] == 1:
                pivot_row = row
                break
        if pivot_row is not None:
            reduced[[rank, pivot_row]] = reduced[[pivot_row, rank]]
            for row in range(number_of_rows):
                if row != rank and reduced[row, column] == 1:
                    reduced[row] ^= reduced[rank]
            pivots.append(column)
            rank += 1
            if rank == number_of_rows:
                break
    free_columns = [column for column in range(number_of_columns) if column not in pivots]
    kernel = np.zeros((len(free_columns), number_of_columns), dtype=np.uint8)
    for kernel_row, free_column in enumerate(free_columns):
        kernel[kernel_row, free_column] = 1
        for pivot_index, pivot_column in enumerate(pivots):
            kernel[kernel_row, pivot_column] = reduced[pivot_index, free_column]
    return kernel


def build_code(name):
    """Return hx, hz and the logical test matrix (kernel of hx) for a code in BB_CODES."""
    l, m, a_terms, b_terms = BB_CODES[name]
    hx, hz = bb_code(l, m, a_terms, b_terms)
    logical_test = gf2_kernel(hx)
    return hx, hz, logical_test
