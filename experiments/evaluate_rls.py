import argparse
import csv
import time

import numpy as np

from src.bp import BeliefPropagation
from src.codes import build_code
from src.sim import decode_all_shots, failures, sample_errors, syndromes
from src.vertical_bp import (
    GreedyQTablePolicy,
    VerticalBeliefPropagation,
    fixed_order,
    random_order,
)


def main():
    parser = argparse.ArgumentParser(description="Task C: RL-S (qubit order) vs baselines.")
    parser.add_argument("--code", default="72")
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--q-table", default="results/rls_q_table.npy")
    parser.add_argument("--output", default="results/task_c_rls.csv")
    settings = parser.parse_args()

    error_rate = settings.error_rate
    hx, hz, logical_test = build_code(settings.code)
    errors = sample_errors(
        hx.shape[1], error_rate, settings.shots, np.random.default_rng(settings.seed)
    )
    all_syndromes = syndromes(hz, errors)
    q_table = np.load(settings.q_table)

    check_decoder = BeliefPropagation(hz, error_rate, settings.max_iterations)
    qubit_decoder = VerticalBeliefPropagation(hz, error_rate, settings.max_iterations)
    schedules = {
        "check_serial": lambda syndrome, rng: check_decoder.decode(syndrome, "serial", rng),
        "check_random_serial": lambda syndrome, rng: check_decoder.decode(
            syndrome, "random_serial", rng
        ),
        "qubit_serial": lambda syndrome, rng: qubit_decoder.decode(syndrome, fixed_order, rng),
        "qubit_random_serial": lambda syndrome, rng: qubit_decoder.decode(
            syndrome, random_order, rng
        ),
        "rl_s": lambda syndrome, rng: qubit_decoder.decode(
            syndrome, GreedyQTablePolicy(q_table), rng
        ),
    }

    all_failed = {}
    with open(settings.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "schedule",
                "shots",
                "ler",
                "standard_error",
                "non_converged",
                "wrong",
                "mean_iterations",
                "seconds",
            ]
        )
        for name, decode_one_shot in schedules.items():
            start_time = time.time()
            guesses, converged, iterations = decode_all_shots(decode_one_shot, all_syndromes)
            failed = failures(hz, logical_test, errors, guesses)
            all_failed[name] = failed
            standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / settings.shots)
            seconds = time.time() - start_time
            writer.writerow(
                [
                    name,
                    settings.shots,
                    failed.mean(),
                    standard_error,
                    (~converged).sum(),
                    (failed & converged).sum(),
                    iterations[converged].mean(),
                    seconds,
                ]
            )
            print(
                f"{name:22s} LER={failed.mean():.4f} +- {standard_error:.4f}  "
                f"non-converged={(~converged).sum():4d}  wrong={(failed & converged).sum():4d}  "
                f"iterations={iterations[converged].mean():.2f}  ({seconds:.0f}s)"
            )

    print()
    for name in ["qubit_serial", "rl_s"]:
        rescued = (all_failed["qubit_random_serial"] & ~all_failed[name]).sum()
        lost = (all_failed[name] & ~all_failed["qubit_random_serial"]).sum()
        print(f"{name:22s} vs qubit_random_serial: rescues {rescued} shots, loses {lost} shots")


if __name__ == "__main__":
    main()
