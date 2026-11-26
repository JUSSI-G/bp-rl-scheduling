import numpy as np
from src.bp import check_to_qubit_messages

FEATURE_NAMES = ["unsatisfied", "light_on", "uncertainty", "wobbling", "residual"]


def all_check_messages(incoming_beliefs, syndrome):
    """Sum-product messages of every check at once (one check per row)."""
    tanh_values = np.tanh(incoming_beliefs / 2)
    products_before = np.concatenate([np.ones((len(tanh_values), 1)), np.cumprod(tanh_values[:, :-1], axis=1)], axis=1)
    reversed_cumulative = np.cumprod(tanh_values[:, ::-1][:, :-1], axis=1)[:, ::-1]
    products_after = np.concatenate([reversed_cumulative, np.ones((len(tanh_values), 1))], axis=1)
    signs = np.where(syndrome == 1, -1.0, 1.0)[:, None]
    combined = np.clip(signs * products_before * products_after, -0.999999999, 0.999999999)
    return 2 * np.arctanh(combined)


class LearnedScheduleDecoder:
    """Serial BP whose next check is picked by a linear softmax policy over check features."""
    def __init__(self, parity_check_matrix, error_rate, weights, max_iterations=50):
        self.parity_check_matrix = parity_check_matrix
        self.number_of_checks, self.number_of_qubits = parity_check_matrix.shape
        self.qubits_in_check = np.array([np.nonzero(row)[0] for row in parity_check_matrix])
        self.prior_belief = np.log((1 - error_rate) / error_rate)
        padded_weights = np.zeros(len(FEATURE_NAMES))
        padded_weights[:len(weights)] = weights
        self.weights = padded_weights
        self.max_iterations = max_iterations

    def check_features(self, beliefs, messages, syndrome, guess_at_start_of_iteration):
        """Features of every check, in the order of FEATURE_NAMES."""
        guess = (beliefs < 0).astype(np.uint8)
        unsatisfied = (self.parity_check_matrix @ guess % 2) != syndrome
        smallest_confidence = np.abs(beliefs)[self.qubits_in_check].min(axis=1)
        uncertainty = 1 / (1 + smallest_confidence)
        changed_qubits = guess != guess_at_start_of_iteration
        wobbling = changed_qubits[self.qubits_in_check].mean(axis=1)
        incoming_beliefs = beliefs[self.qubits_in_check] - messages
        new_messages = all_check_messages(incoming_beliefs, syndrome)
        largest_change = np.abs(new_messages - messages).max(axis=1)
        residual = largest_change / (1 + largest_change)
        return np.column_stack([unsatisfied, syndrome, uncertainty, wobbling, residual]).astype(float)

    def choose_check(self, features, remaining, rng, explore):
        """Pick the next check; when exploring, also return the log-policy gradient."""
        scores = features[remaining] @ self.weights
        if explore:
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            position = rng.choice(len(remaining), p=probabilities)
            gradient = features[remaining][position] - probabilities @ features[remaining]
            return remaining[position], gradient
        best = np.flatnonzero(scores == scores.max())
        return remaining[rng.choice(best)], None

    def decode(self, syndrome, rng, explore=False):
        """Return the guess, convergence, iterations and the summed policy gradient."""
        beliefs = np.full(self.number_of_qubits, self.prior_belief)
        messages = np.zeros(self.qubits_in_check.shape)
        total_gradient = np.zeros_like(self.weights)

        for iteration in range(1, self.max_iterations + 1):
            guess_at_start_of_iteration = (beliefs < 0).astype(np.uint8)
            remaining = list(range(self.number_of_checks))
            while remaining:
                features = self.check_features(beliefs, messages, syndrome, guess_at_start_of_iteration)
                check, gradient = self.choose_check(features, np.array(remaining), rng, explore)
                if gradient is not None:
                    total_gradient += gradient
                remaining.remove(check)

                qubits = self.qubits_in_check[check]
                incoming_beliefs = beliefs[qubits] - messages[check]
                messages[check] = check_to_qubit_messages(incoming_beliefs, syndrome[check])
                beliefs[qubits] = incoming_beliefs + messages[check]

            guess = (beliefs < 0).astype(np.uint8)
            if np.array_equal(self.parity_check_matrix @ guess % 2, syndrome):
                return guess, True, iteration, total_gradient

        return guess, False, self.max_iterations, total_gradient