import csv
import json
import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation
from src.rl import LearnedScheduleDecoder

l, m = 6, 6
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
error_rate = 0.05
number_of_shots = 2000
max_iterations = 50
error_seed = 1

hx, hz = bb_code(l, m, a_terms, b_terms)
logical_test = gf2_kernel(hx)
errors = sample_errors(hx.shape[1], error_rate, number_of_shots, np.random.default_rng(error_seed))
all_syndromes = syndromes(hz, errors)
learned_weights = json.load(open("results/learned_weights.json"))["weights"]


def evaluate(decode_one_shot):
    start_time = time.time()
    results = [decode_one_shot(syndrome, np.random.default_rng([99, shot]))
               for shot, syndrome in enumerate(all_syndromes)]
    guesses = np.array([result[0] for result in results], dtype=np.uint8)
    converged = np.array([result[1] for result in results])
    iterations = np.array([result[2] for result in results])
    failed = failures(hz, logical_test, errors, guesses)
    return failed, converged, iterations, time.time() - start_time


plain_decoder = BeliefPropagation(hz, error_rate, max_iterations)
schedules = {
    "flooding": lambda syndrome, rng: plain_decoder.decode(syndrome, "flooding", rng),
    "serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "serial", rng),
    "random_serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "random_serial", rng),
    "rule_unsatisfied_first": LearnedScheduleDecoder(hz, error_rate, [5, 0, 0, 0], max_iterations).decode,
    "rule_confident_first": LearnedScheduleDecoder(hz, error_rate, [0, 0, -5, 0], max_iterations).decode,
    "learned": LearnedScheduleDecoder(hz, error_rate, learned_weights, max_iterations).decode,
}

all_failed = {}
with open("results/task_c.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["schedule", "shots", "ler", "standard_error", "non_converged", "wrong",
                     "mean_iterations", "seconds"])
    for name, decode_one_shot in schedules.items():
        failed, converged, iterations, seconds = evaluate(decode_one_shot)
        all_failed[name] = failed
        standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / number_of_shots)
        writer.writerow([name, number_of_shots, failed.mean(), standard_error, (~converged).sum(),
                         (failed & converged).sum(), iterations[converged].mean(), seconds])
        print(f"{name:24s} LER={failed.mean():.4f} ± {standard_error:.4f}  "
              f"non-converged={(~converged).sum():3d}  wrong={(failed & converged).sum():3d}  "
              f"iterations={iterations[converged].mean():.2f}  ({seconds:.0f}s)")

print()
for name in ["serial", "rule_unsatisfied_first", "rule_confident_first", "learned"]:
    only_random_failed = (all_failed["random_serial"] & ~all_failed[name]).sum()
    only_this_failed = (all_failed[name] & ~all_failed["random_serial"]).sum()
    print(f"{name:24s} vs random_serial: rescues {only_random_failed} shots, loses {only_this_failed} shots")