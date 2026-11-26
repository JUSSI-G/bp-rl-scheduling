import argparse
import csv
import json
import time

import numpy as np

from src.bp import BeliefPropagation
from src.codes import BB_CODES, build_code
from src.rl import LearnedScheduleDecoder
from src.sim import decode_all_shots, failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(description="Evaluate learned weight files against baselines.")
    parser.add_argument("--code", default="144", choices=BB_CODES.keys())
    parser.add_argument("--error-rates", type=float, nargs="+", default=[0.05])
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    settings = parser.parse_args()

    hx, hz, logical_test = build_code(settings.code)

    with open(settings.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "code",
                "error_rate",
                "schedule",
                "shots",
                "max_iterations",
                "ler",
                "standard_error",
                "non_converged",
                "wrong",
                "mean_iterations",
                "rescued_vs_random",
                "lost_vs_random",
                "seconds",
            ]
        )
        for error_rate in settings.error_rates:
            errors = sample_errors(
                hx.shape[1], error_rate, settings.shots, np.random.default_rng(settings.seed)
            )
            all_syndromes = syndromes(hz, errors)
            plain_decoder = BeliefPropagation(hz, error_rate, settings.max_iterations)
            schedules = {
                "flooding": lambda syndrome, rng: plain_decoder.decode(syndrome, "flooding", rng),
                "serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "serial", rng),
                "random_serial": lambda syndrome, rng: plain_decoder.decode(
                    syndrome, "random_serial", rng
                ),
            }
            for weights_file in settings.weights:
                with open(weights_file) as weights_handle:
                    weights = json.load(weights_handle)["weights"]
                agent = LearnedScheduleDecoder(hz, error_rate, weights, settings.max_iterations)
                schedules[weights_file] = agent.decode

            random_failed = None
            for name, decode_one_shot in schedules.items():
                start_time = time.time()
                guesses, converged, iterations = decode_all_shots(decode_one_shot, all_syndromes)
                failed = failures(hz, logical_test, errors, guesses)
                if name == "random_serial":
                    random_failed = failed
                rescued = (random_failed & ~failed).sum() if random_failed is not None else ""
                lost = (failed & ~random_failed).sum() if random_failed is not None else ""
                standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / settings.shots)
                seconds = time.time() - start_time
                writer.writerow(
                    [
                        settings.code,
                        error_rate,
                        name,
                        settings.shots,
                        settings.max_iterations,
                        failed.mean(),
                        standard_error,
                        (~converged).sum(),
                        (failed & converged).sum(),
                        iterations[converged].mean(),
                        rescued,
                        lost,
                        seconds,
                    ]
                )
                file.flush()
                print(
                    f"[[{settings.code}]] p={error_rate:.3f} {name:40s} "
                    f"LER={failed.mean():.4f} +- {standard_error:.4f}  "
                    f"stuck={(~converged).sum():4d}  wrong={(failed & converged).sum():4d}  "
                    f"iters={iterations[converged].mean():.2f}  vs random: +{rescued} -{lost}  "
                    f"({seconds:.0f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
