import numpy as np


def product_of_others(values):
    """Product of all other entries, for every position."""
    products_before = np.concatenate(([1.0], np.cumprod(values[:-1])))
    products_after = np.concatenate((np.cumprod(values[::-1][:-1])[::-1], [1.0]))
    return products_before * products_after


def check_to_qubit_messages(incoming_beliefs, syndrome_bit):
    """Sum-product messages from one check to each of its qubits."""
    tanh_values = np.tanh(incoming_beliefs / 2)
    sign = -1.0 if syndrome_bit == 1 else 1.0
    combined = sign * product_of_others(tanh_values)
    combined = np.clip(combined, -0.999999999, 0.999999999)
    return 2 * np.arctanh(combined)


class BeliefPropagation:
    """Sum-product BP decoder with flooding, serial and random serial schedules."""

    def __init__(self, parity_check_matrix, error_rate, max_iterations=50):
        self.parity_check_matrix = parity_check_matrix
        self.number_of_checks, self.number_of_qubits = parity_check_matrix.shape
        self.qubits_in_check = [np.nonzero(row)[0] for row in parity_check_matrix]
        self.prior_belief = np.log((1 - error_rate) / error_rate)
        self.max_iterations = max_iterations

    def guess_matches_syndrome(self, guess, syndrome):
        """Whether the guess reproduces the measured syndrome."""
        return np.array_equal(self.parity_check_matrix @ guess % 2, syndrome)

    def flooding_iteration(self, beliefs, messages, syndrome):
        """Update all checks at once from the previous beliefs."""
        new_messages = []
        for check, qubits in enumerate(self.qubits_in_check):
            incoming_beliefs = beliefs[qubits] - messages[check]
            new_messages.append(check_to_qubit_messages(incoming_beliefs, syndrome[check]))
        new_beliefs = np.full(self.number_of_qubits, self.prior_belief)
        for check, qubits in enumerate(self.qubits_in_check):
            new_beliefs[qubits] += new_messages[check]
        return new_beliefs, new_messages

    def serial_iteration(self, beliefs, messages, syndrome, check_order):
        """Update the checks one by one in the given order."""
        for check in check_order:
            qubits = self.qubits_in_check[check]
            incoming_beliefs = beliefs[qubits] - messages[check]
            messages[check] = check_to_qubit_messages(incoming_beliefs, syndrome[check])
            beliefs[qubits] = incoming_beliefs + messages[check]
        return beliefs, messages

    def decode(self, syndrome, schedule="flooding", rng=None):
        """Return the guess, whether BP converged, and the number of iterations."""
        beliefs = np.full(self.number_of_qubits, self.prior_belief)
        messages = [np.zeros(len(qubits)) for qubits in self.qubits_in_check]
        fixed_order = np.arange(self.number_of_checks)

        for iteration in range(1, self.max_iterations + 1):
            if schedule == "flooding":
                beliefs, messages = self.flooding_iteration(beliefs, messages, syndrome)
            elif schedule == "serial":
                beliefs, messages = self.serial_iteration(beliefs, messages, syndrome, fixed_order)
            elif schedule == "random_serial":
                random_order = rng.permutation(self.number_of_checks)
                beliefs, messages = self.serial_iteration(beliefs, messages, syndrome, random_order)

            guess = (beliefs < 0).astype(np.uint8)
            if self.guess_matches_syndrome(guess, syndrome):
                return guess, True, iteration

        return guess, False, self.max_iterations
