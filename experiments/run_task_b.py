import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation

code_name = "72_12_6"
l, m = 6, 6
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
error_rate = 0.05
number_of_shots = 1000
number_of_seeds = 20
max_iterations = 50
error_seed = 1

hx, hz = bb_code(l, m, a_terms, b_terms)
number_of_qubits = hx.shape[1]
logical_test = gf2_kernel(hx)

error_rng = np.random.default_rng(error_seed)
errors = sample_errors(number_of_qubits, error_rate, number_of_shots, error_rng)
all_syndromes = syndromes(hz, errors)
decoder = BeliefPropagation(hz, error_rate, max_iterations)


def decode_all(schedule, seed):
    guesses = []
    for shot, syndrome in enumerate(all_syndromes):
        shot_rng = np.random.default_rng([seed, shot])
        guess, converged, iterations = decoder.decode(syndrome, schedule, shot_rng)
        guesses.append(guess)
    return failures(hz, logical_test, errors, np.array(guesses, dtype=np.uint8))


start_time = time.time()
flooding_failed = decode_all("flooding", 0)
serial_failed = decode_all("serial", 0)
random_failed = np.zeros((number_of_shots, number_of_seeds), dtype=bool)
for seed in range(number_of_seeds):
    random_failed[:, seed] = decode_all("random_serial", seed)
    print(f"seed {seed:2d}: LER={random_failed[:, seed].mean():.4f}")

np.savez(f"results/task_b_{code_name}.npz", flooding_failed=flooding_failed,
         serial_failed=serial_failed, random_failed=random_failed,
         error_rate=error_rate, error_seed=error_seed, max_iterations=max_iterations)

seed_lers = random_failed.mean(axis=0)
always_fail = random_failed.all(axis=1)
never_fail = (~random_failed).all(axis=1)
sometimes_fail = ~always_fail & ~never_fail

print()
print(f"flooding LER         {flooding_failed.mean():.4f}")
print(f"serial LER           {serial_failed.mean():.4f}")
print(f"random serial LER    mean {seed_lers.mean():.4f}  std {seed_lers.std():.4f}  "
      f"min {seed_lers.min():.4f}  max {seed_lers.max():.4f}")
print(f"oracle LER (any of {number_of_seeds} seeds succeeds)  {always_fail.mean():.4f}")
print(f"shots: always succeed {never_fail.sum()}, always fail {always_fail.sum()}, "
      f"depends on order {sometimes_fail.sum()}")
print(f"time {time.time() - start_time:.0f}s")