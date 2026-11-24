import numpy as np
from src.bp import check_to_qubit_messages

feature_names = ["unsatisfied", "light_on", "uncertainty", "wobbling"]


class LearnedScheduleDecoder:
    def __init__(self, parity_check_matrix, error_rate, weights, max_iterations=50):
        self.parity_check_matrix = parity_check_matrix
        self.number_of_checks, self.number_of_qubits = parity_check_matrix.shape
        self.qubits_in_check = np.array([np.nonzero(row)[0] for row in parity_check_matrix])
        self.prior_belief = np.log((1 - error_rate) / error_rate)
        self.weights = np.array(weights, dtype=float)
        self.max_iterations = max_iterations

    def check_features(self, beliefs, syndrome, guess_at_start_of_iteration):
        guess = (beliefs < 0).astype(np.uint8)
        unsatisfied = (self.parity_check_matrix @ guess % 2) != syndrome
        smallest_confidence = np.abs(beliefs)[self.qubits_in_check].min(axis=1)
        uncertainty = 1 / (1 + smallest_confidence)
        changed_qubits = guess != guess_at_start_of_iteration
        wobbling = changed_qubits[self.qubits_in_check].mean(axis=1)
        return np.column_stack([unsatisfied, syndrome, uncertainty, wobbling]).astype(float)

    def choose_check(self, features, remaining, rng, explore):
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
        beliefs = np.full(self.number_of_qubits, self.prior_belief)
        messages = np.zeros(self.qubits_in_check.shape)
        total_gradient = np.zeros_like(self.weights)

        for iteration in range(1, self.max_iterations + 1):
            guess_at_start_of_iteration = (beliefs < 0).astype(np.uint8)
            remaining = list(range(self.number_of_checks))
            while remaining:
                features = self.check_features(beliefs, syndrome, guess_at_start_of_iteration)
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