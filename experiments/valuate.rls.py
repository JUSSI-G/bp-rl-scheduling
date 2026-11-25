import csv
import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation
from src.vertical_bp import VerticalBeliefPropagation, fixed_order, random_order, GreedyQTablePolicy

l, m = 6, 6
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
error_rate = 0.05
number_of_shots = 10000
max_iterations = 50
error_seed = 1

hx, hz = bb_code(l, m, a_terms, b_terms)
logical_test = gf2_kernel(hx)
errors = sample_errors(hx.shape[1], error_rate, number_of_shots, np.random.default_rng(error_seed))
all_syndromes = syndromes(hz, errors)
q_table = np.load("results/rls_q_table.npy")

check_decoder = BeliefPropagation(hz, error_rate, max_iterations)
qubit_decoder = VerticalBeliefPropagation(hz, error_rate, max_iterations)
schedules = {
    "check_serial": lambda syndrome, rng: check_decoder.decode(syndrome, "serial", rng),
    "check_random_serial": lambda syndrome, rng: check_decoder.decode(syndrome, "random_serial", rng),
    "qubit_serial": lambda syndrome, rng: qubit_decoder.decode(syndrome, fixed_order, rng),
    "qubit_random_serial": lambda syndrome, rng: qubit_decoder.decode(syndrome, random_order, rng),
    "rl_s": lambda syndrome, rng: qubit_decoder.decode(syndrome, GreedyQTablePolicy(q_table), rng),
}

all_failed = {}
with open("results/task_c_rls.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["schedule", "shots", "ler", "standard_error", "non_converged", "wrong",
                     "mean_iterations", "seconds"])
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
        writer.writerow([name, number_of_shots, failed.mean(), standard_error, (~converged).sum(),
                         (failed & converged).sum(), iterations[converged].mean(), seconds])
        print(f"{name:22s} LER={failed.mean():.4f} +- {standard_error:.4f}  "
              f"non-converged={(~converged).sum():4d}  wrong={(failed & converged).sum():4d}  "
              f"iterations={iterations[converged].mean():.2f}  ({seconds:.0f}s)")

print()
for name in ["qubit_serial", "rl_s"]:
    rescued = (all_failed["qubit_random_serial"] & ~all_failed[name]).sum()
    lost = (all_failed[name] & ~all_failed["qubit_random_serial"]).sum()
    print(f"{name:22s} vs qubit_random_serial: rescues {rescued} shots, loses {lost} shots")