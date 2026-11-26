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
    parser = argparse.ArgumentParser(description="Task E: apply the rule learned on 72 to 144.")
    parser.add_argument("--code", default="144")
    parser.add_argument("--error-rates", type=float, nargs="+", default=[0.04, 0.05])
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", default="results/learned_weights_long.json")
    parser.add_argument("--output", default="results/task_e_transfer_144.csv")
    settings = parser.parse_args()

    max_iterations = settings.max_iterations
    hx, hz, logical_test = build_code(settings.code)
    with open(settings.weights) as file:
        learned_weights = json.load(file)["weights"]

    with open(settings.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "error_rate",
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
        for error_rate in settings.error_rates:
            errors = sample_errors(
                hx.shape[1], error_rate, settings.shots, np.random.default_rng(settings.seed)
            )
            all_syndromes = syndromes(hz, errors)
            plain_decoder = BeliefPropagation(hz, error_rate, max_iterations)
            schedules = {
                "flooding": lambda syndrome, rng: plain_decoder.decode(syndrome, "flooding", rng),
                "serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "serial", rng),
                "random_serial": lambda syndrome, rng: plain_decoder.decode(
                    syndrome, "random_serial", rng
                ),
                "learned_on_72": LearnedScheduleDecoder(
                    hz, error_rate, learned_weights, max_iterations
                ).decode,
            }
            all_failed = {}
            for name, decode_one_shot in schedules.items():
                start_time = time.time()
                guesses, converged, iterations = decode_all_shots(decode_one_shot, all_syndromes)
                failed = failures(hz, logical_test, errors, guesses)
                all_failed[name] = failed
                standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / settings.shots)
                seconds = time.time() - start_time
                writer.writerow(
                    [
                        error_rate,
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
                file.flush()
                print(
                    f"p={error_rate:.2f} {name:15s} "
                    f"LER={failed.mean():.4f} +- {standard_error:.4f}  "
                    f"non-converged={(~converged).sum():4d}  "
                    f"wrong={(failed & converged).sum():4d}  "
                    f"iterations={iterations[converged].mean():.2f}  ({seconds:.0f}s)"
                )
            rescued = (all_failed["random_serial"] & ~all_failed["learned_on_72"]).sum()
            lost = (all_failed["learned_on_72"] & ~all_failed["random_serial"]).sum()
            print(
                f"p={error_rate:.2f} learned_on_72 vs random_serial: "
                f"rescues {rescued}, loses {lost}"
            )
            print()


if __name__ == "__main__":
    main()
