import json

import numpy as np
import pytest
from ldpc import BpDecoder

from src.bp import BeliefPropagation
from src.codes import build_code
from src.rl import LearnedScheduleDecoder
from src.sim import sample_errors, syndromes
from src.vertical_bp import VerticalBeliefPropagation, random_order

ERROR_RATE = 0.05
SHOTS = 200


@pytest.fixture(scope="module")
def code_and_syndromes():
    hx, hz, logical_test = build_code("72")
    errors = sample_errors(hx.shape[1], ERROR_RATE, SHOTS, np.random.default_rng(1))
    return hz, syndromes(hz, errors)


def test_flooding_matches_ldpc(code_and_syndromes):
    hz, all_syndromes = code_and_syndromes
    library_decoder = BpDecoder(
        hz, error_rate=ERROR_RATE, max_iter=50, bp_method="product_sum", schedule="parallel"
    )
    own_decoder = BeliefPropagation(hz, ERROR_RATE, 50)
    for syndrome in all_syndromes:
        own_guess = own_decoder.decode(syndrome, "flooding")[0]
        assert np.array_equal(own_guess, library_decoder.decode(syndrome))


@pytest.mark.parametrize("schedule", ["serial", "random_serial"])
def test_converged_serial_guess_satisfies_syndrome(code_and_syndromes, schedule):
    hz, all_syndromes = code_and_syndromes
    decoder = BeliefPropagation(hz, ERROR_RATE, 50)
    rng = np.random.default_rng(7)
    converged_shots = 0
    for syndrome in all_syndromes:
        guess, converged, iterations = decoder.decode(syndrome, schedule, rng)
        if converged:
            converged_shots += 1
            assert np.array_equal(hz @ guess % 2, syndrome)
    assert converged_shots > SHOTS // 2


def test_converged_learned_and_vertical_guesses_satisfy_syndrome(code_and_syndromes):
    hz, all_syndromes = code_and_syndromes
    with open("results/learned_weights_long.json") as file:
        weights = json.load(file)["weights"]
    learned_decoder = LearnedScheduleDecoder(hz, ERROR_RATE, weights, 20)
    vertical_decoder = VerticalBeliefPropagation(hz, ERROR_RATE, 20)
    rng = np.random.default_rng(7)
    for syndrome in all_syndromes[:30]:
        guess, converged, iterations, gradient = learned_decoder.decode(syndrome, rng)
        if converged:
            assert np.array_equal(hz @ guess % 2, syndrome)
        guess, converged, iterations = vertical_decoder.decode(syndrome, random_order, rng)
        if converged:
            assert np.array_equal(hz @ guess % 2, syndrome)
