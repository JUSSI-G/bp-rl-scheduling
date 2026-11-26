import argparse
import csv
import time

import numpy as np

from src.bp import BeliefPropagation
from src.codes import build_code
from src.sim import failures, sample_errors, syndromes

CSV_CODE_NAMES = {"72": "72_12_6", "144": "144_12_12", "288": "288_12_18"}
SCHEDULES = ["flooding", "serial", "random_serial"]


def main():
    parser = argparse.ArgumentParser(description="Task A: flooding vs serial vs random serial BP.")
    parser.add_argument("--codes", nargs="+", default=["72", "144"], choices=CSV_CODE_NAMES.keys())
    parser.add_argument(
        "--error-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
    )
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--schedule-seed", type=int, default=7)
    parser.add_argument("--output", default="results/task_a.csv")
    settings = parser.parse_args()

    columns = [
        "code",
        "error_rate",
        "schedule",
        "shots",
        "failures",
        "non_converged",
        "wrong",
        "ler",
        "mean_iterations",
        "max_iterations",
        "error_seed",
        "schedule_seed",
    ]
    with open(settings.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for code in settings.codes:
            code_name = CSV_CODE_NAMES[code]
            hx, hz, logical_test = build_code(code)
            number_of_qubits = hx.shape[1]

            for error_rate in settings.error_rates:
                error_rng = np.random.default_rng(settings.seed)
                errors = sample_errors(number_of_qubits, error_rate, settings.shots, error_rng)
                all_syndromes = syndromes(hz, errors)

                for schedule in SCHEDULES:
                    decoder = BeliefPropagation(hz, error_rate, settings.max_iterations)
                    schedule_rng = np.random.default_rng(settings.schedule_seed)
                    start_time = time.time()
                    results = [decoder.decode(s, schedule, schedule_rng) for s in all_syndromes]
                    guesses = np.array([result[0] for result in results], dtype=np.uint8)
                    converged = np.array([result[1] for result in results])
                    iterations = np.array([result[2] for result in results])
                    failed = failures(hz, logical_test, errors, guesses)

                    writer.writerow(
                        [
                            code_name,
                            error_rate,
                            schedule,
                            settings.shots,
                            failed.sum(),
                            (~converged).sum(),
                            (failed & converged).sum(),
                            failed.mean(),
                            iterations[converged].mean(),
                            settings.max_iterations,
                            settings.seed,
                            settings.schedule_seed,
                        ]
                    )
                    file.flush()
                    print(
                        f"{code_name:10s} p={error_rate:.2f} {schedule:14s} "
                        f"LER={failed.mean():.4f} iterations={iterations[converged].mean():.2f} "
                        f"({time.time() - start_time:.0f}s)"
                    )


if __name__ == "__main__":
    main()
