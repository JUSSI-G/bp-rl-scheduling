import argparse
import csv
import json
import time

import numpy as np

from src.bp import BeliefPropagation
from src.codes import build_code
from src.rl import LearnedScheduleDecoder
from src.sim import decode_all_shots, failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(description="Task C: learned check schedule vs baselines.")
    parser.add_argument("--code", default="72")
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", default="results/learned_weights_long.json")
    parser.add_argument("--output", default="results/task_c_long.csv")
    settings = parser.parse_args()

    error_rate = settings.error_rate
    max_iterations = settings.max_iterations
    hx, hz, logical_test = build_code(settings.code)
    errors = sample_errors(
        hx.shape[1], error_rate, settings.shots, np.random.default_rng(settings.seed)
    )
    all_syndromes = syndromes(hz, errors)
    with open(settings.weights) as file:
        learned_weights = json.load(file)["weights"]

    plain_decoder = BeliefPropagation(hz, error_rate, max_iterations)
    schedules = {
        "flooding": lambda syndrome, rng: plain_decoder.decode(syndrome, "flooding", rng),
        "serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "serial", rng),
        "random_serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "random_serial", rng),
        "rule_unsatisfied_first": LearnedScheduleDecoder(
            hz, error_rate, [5, 0, 0, 0], max_iterations
        ).decode,
        "rule_confident_first": LearnedScheduleDecoder(
            hz, error_rate, [0, 0, -5, 0], max_iterations
        ).decode,
        "learned": LearnedScheduleDecoder(hz, error_rate, learned_weights, max_iterations).decode,
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
            seconds = time.time() - start_time
            all_failed[name] = failed
            standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / settings.shots)
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
                f"{name:24s} LER={failed.mean():.4f} +- {standard_error:.4f}  "
                f"non-converged={(~converged).sum():3d}  wrong={(failed & converged).sum():3d}  "
                f"iterations={iterations[converged].mean():.2f}  ({seconds:.0f}s)"
            )

    print()
    for name in ["serial", "rule_unsatisfied_first", "rule_confident_first", "learned"]:
        rescued = (all_failed["random_serial"] & ~all_failed[name]).sum()
        lost = (all_failed[name] & ~all_failed["random_serial"]).sum()
        print(f"{name:24s} vs random_serial: rescues {rescued} shots, loses {lost} shots")


if __name__ == "__main__":
    main()
