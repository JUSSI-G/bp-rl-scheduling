import csv
import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation

codes = {
    "72_12_6": (6, 6),
    "144_12_12": (12, 6),
}
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
error_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
schedules = ["flooding", "serial", "random_serial"]
number_of_shots = 2000
max_iterations = 50
error_seed = 1
schedule_seed = 7

output_file = "results/task_a.csv"
columns = ["code", "error_rate", "schedule", "shots", "failures", "non_converged",
           "wrong", "ler", "mean_iterations", "max_iterations", "error_seed", "schedule_seed"]

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)

    for code_name, (l, m) in codes.items():
        hx, hz = bb_code(l, m, a_terms, b_terms)
        number_of_qubits = hx.shape[1]
        logical_test = gf2_kernel(hx)

        for error_rate in error_rates:
            error_rng = np.random.default_rng(error_seed)
            errors = sample_errors(number_of_qubits, error_rate, number_of_shots, error_rng)
            all_syndromes = syndromes(hz, errors)

            for schedule in schedules:
                decoder = BeliefPropagation(hz, error_rate, max_iterations)
                schedule_rng = np.random.default_rng(schedule_seed)
                start_time = time.time()
                results = [decoder.decode(s, schedule, schedule_rng) for s in all_syndromes]
                guesses = np.array([guess for guess, converged, iterations in results], dtype=np.uint8)
                converged = np.array([converged for guess, converged, iterations in results])
                iterations = np.array([iterations for guess, converged, iterations in results])
                failed = failures(hz, logical_test, errors, guesses)

                writer.writerow([code_name, error_rate, schedule, number_of_shots, failed.sum(),
                                 (~converged).sum(), (failed & converged).sum(), failed.mean(),
                                 iterations[converged].mean(), max_iterations,
                                 error_seed, schedule_seed])
                file.flush()
                print(f"{code_name:10s} p={error_rate:.2f} {schedule:14s} "
                      f"LER={failed.mean():.4f} iterations={iterations[converged].mean():.2f} "
                      f"({time.time() - start_time:.0f}s)")