import numpy as np


def sample_errors(number_of_qubits, error_rate, shots, rng):
    """Independent X errors on every qubit with probability error_rate."""
    random_values = rng.random((shots, number_of_qubits))
    return (random_values < error_rate).astype(np.uint8)


def syndromes(parity_check_matrix, errors):
    """Syndrome of every error (one per row)."""
    return errors @ parity_check_matrix.T % 2


def failures(parity_check_matrix, logical_test, errors, guesses):
    """True for every shot whose guess leaves a nonzero syndrome or a logical error."""
    residual_errors = errors ^ guesses
    not_converged = (residual_errors @ parity_check_matrix.T % 2).any(axis=1)
    logical_error = (residual_errors @ logical_test.T % 2).any(axis=1)
    return not_converged | logical_error


def decode_all_shots(decode_one_shot, all_syndromes):
    """Decode every syndrome with its own generator seeded by [99, shot]."""
    results = [
        decode_one_shot(syndrome, np.random.default_rng([99, shot]))
        for shot, syndrome in enumerate(all_syndromes)
    ]
    guesses = np.array([result[0] for result in results], dtype=np.uint8)
    converged = np.array([result[1] for result in results])
    iterations = np.array([result[2] for result in results])
    return guesses, converged, iterations
