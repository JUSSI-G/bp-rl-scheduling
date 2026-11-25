import csv
import json
import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation
from src.rl import LearnedScheduleDecoder

l, m = 12, 6
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
error_rates = [0.04, 0.05]
number_of_shots = 5000
max_iterations = 50
error_seed = 1

hx, hz = bb_code(l, m, a_terms, b_terms)
logical_test = gf2_kernel(hx)
learned_weights = json.load(open("results/learned_weights_long.json"))["weights"]

with open("results/task_e_transfer_144.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["error_rate", "schedule", "shots", "ler", "standard_error", "non_converged",
                     "wrong", "mean_iterations", "seconds"])
    for error_rate in error_rates:
        errors = sample_errors(hx.shape[1], error_rate, number_of_shots, np.random.default_rng(error_seed))
        all_syndromes = syndromes(hz, errors)
        plain_decoder = BeliefPropagation(hz, error_rate, max_iterations)
        schedules = {
            "flooding": lambda syndrome, rng: plain_decoder.decode(syndrome, "flooding", rng),
            "serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "serial", rng),
            "random_serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "random_serial", rng),
            "learned_on_72": LearnedScheduleDecoder(hz, error_rate, learned_weights, max_iterations).decode,
        }
        all_failed = {}
        for name, decode_one_shot in schedules.items():
            start_time = time.time()
            results = [decode_one_shot(syndrome, np.random.default_rng([99, shot]))
                       for shot, syndrome in enumerate(all_syndromes)]
            guesses = np.array([result[0] for result in results], dtype=np.uint8)
            converged = np.array([result[1] for result in results])
            iterations = np.array([result[2] for result in results])
            failed = failures(hz, logical_test, errors, guesses)
            all_failed[name] = failed
            standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / number_of_shots)
            seconds = time.time() - start_time
            writer.writerow([error_rate, name, number_of_shots, failed.mean(), standard_error,
                             (~converged).sum(), (failed & converged).sum(),
                             iterations[converged].mean(), seconds])
            file.flush()
            print(f"p={error_rate:.2f} {name:15s} LER={failed.mean():.4f} +- {standard_error:.4f}  "
                  f"non-converged={(~converged).sum():4d}  wrong={(failed & converged).sum():4d}  "
                  f"iterations={iterations[converged].mean():.2f}  ({seconds:.0f}s)")
        rescued = (all_failed["random_serial"] & ~all_failed["learned_on_72"]).sum()
        lost = (all_failed["learned_on_72"] & ~all_failed["random_serial"]).sum()
        print(f"p={error_rate:.2f} learned_on_72 vs random_serial: rescues {rescued}, loses {lost}")
        print()