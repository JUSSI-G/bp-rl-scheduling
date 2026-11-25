import numpy as np


def tanh_product_message(other_messages, syndrome_bit):
    sign = -1.0 if syndrome_bit == 1 else 1.0
    combined = sign * np.prod(np.tanh(other_messages / 2))
    combined = np.clip(combined, -0.999999999, 0.999999999)
    return 2 * np.arctanh(combined)


class VerticalBeliefPropagation:
    def __init__(self, parity_check_matrix, error_rate, max_iterations=50):
        self.parity_check_matrix = parity_check_matrix
        self.number_of_checks, self.number_of_qubits = parity_check_matrix.shape
        self.qubits_in_check = np.array([np.nonzero(row)[0] for row in parity_check_matrix])
        self.checks_of_qubit = np.array([np.nonzero(column)[0] for column in parity_check_matrix.T])
        self.position_in_check = np.array([
            [list(self.qubits_in_check[check]).index(qubit) for check in self.checks_of_qubit[qubit]]
            for qubit in range(self.number_of_qubits)
        ])
        self.prior_belief = np.log((1 - error_rate) / error_rate)
        self.max_iterations = max_iterations

    def start(self, syndrome):
        self.syndrome = syndrome
        self.qubit_to_check = np.full(self.qubits_in_check.shape, self.prior_belief)
        self.guess = np.zeros(self.number_of_qubits, dtype=np.uint8)
        self.unsatisfied = syndrome.astype(bool).copy()

    def update_qubit(self, qubit):
        checks = self.checks_of_qubit[qubit]
        positions = self.position_in_check[qubit]
        check_to_qubit = np.empty(len(checks))
        for k, (check, position) in enumerate(zip(checks, positions)):
            other_messages = np.delete(self.qubit_to_check[check], position)
            check_to_qubit[k] = tanh_product_message(other_messages, self.syndrome[check])
        belief = self.prior_belief + check_to_qubit.sum()
        self.qubit_to_check[checks, positions] = belief - check_to_qubit
        new_guess = 1 if belief < 0 else 0
        if new_guess != self.guess[qubit]:
            self.guess[qubit] = new_guess
            self.unsatisfied[checks] = ~self.unsatisfied[checks]

    def qubit_states(self):
        bit_values = 2 ** np.arange(self.checks_of_qubit.shape[1])
        return (self.unsatisfied[self.checks_of_qubit] * bit_values).sum(axis=1)

    def decode(self, syndrome, choose_qubit, rng):
        self.start(syndrome)
        for iteration in range(1, self.max_iterations + 1):
            remaining = np.ones(self.number_of_qubits, dtype=bool)
            for step in range(self.number_of_qubits):
                if not self.unsatisfied.any():
                    return self.guess.copy(), True, iteration
                qubit = choose_qubit(self, remaining, iteration, step, rng)
                remaining[qubit] = False
                self.update_qubit(qubit)
        converged = not self.unsatisfied.any()
        return self.guess.copy(), converged, self.max_iterations


def fixed_order(decoder, remaining, iteration, step, rng):
    return np.flatnonzero(remaining)[0]


def random_order(decoder, remaining, iteration, step, rng):
    return rng.choice(np.flatnonzero(remaining))


class GreedyQTablePolicy:
    def __init__(self, q_table):
        self.q_table = q_table

    def __call__(self, decoder, remaining, iteration, step, rng):
        candidates = np.flatnonzero(remaining)
        states = decoder.qubit_states()[candidates]
        values = self.q_table[states, candidates]
        return candidates[np.argmax(values)]