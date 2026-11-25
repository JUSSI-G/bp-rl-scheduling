import argparse
import csv
import json
import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation
from src.rl import LearnedScheduleDecoder

codes = {"72": (6, 6), "144": (12, 6)}
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]

parser = argparse.ArgumentParser()
parser.add_argument("--code", default="144", choices=codes.keys())
parser.add_argument("--error-rates", type=float, nargs="+", default=[0.05])
parser.add_argument("--shots", type=int, default=5000)
parser.add_argument("--max-iterations", type=int, default=50)
parser.add_argument("--weights", nargs="+", required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", required=True)
settings = parser.parse_args()

l, m = codes[settings.code]
hx, hz = bb_code(l, m, a_terms, b_terms)
logical_test = gf2_kernel(hx)

with open(settings.output, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["code", "error_rate", "schedule", "shots", "max_iterations", "ler", "standard_error",
                     "non_converged", "wrong", "mean_iterations", "rescued_vs_random",
                     "lost_vs_random", "seconds"])
    for error_rate in settings.error_rates:
        errors = sample_errors(hx.shape[1], error_rate, settings.shots, np.random.default_rng(settings.seed))
        all_syndromes = syndromes(hz, errors)
        plain_decoder = BeliefPropagation(hz, error_rate, settings.max_iterations)
        schedules = {
            "flooding": lambda syndrome, rng: plain_decoder.decode(syndrome, "flooding", rng),
            "serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "serial", rng),
            "random_serial": lambda syndrome, rng: plain_decoder.decode(syndrome, "random_serial", rng),
        }
        for weights_file in settings.weights:
            weights = json.load(open(weights_file))["weights"]
            agent = LearnedScheduleDecoder(hz, error_rate, weights, settings.max_iterations)
            schedules[weights_file] = agent.decode

        random_failed = None
        for name, decode_one_shot in schedules.items():
            start_time = time.time()
            results = [decode_one_shot(syndrome, np.random.default_rng([99, shot]))
                       for shot, syndrome in enumerate(all_syndromes)]
            guesses = np.array([result[0] for result in results], dtype=np.uint8)
            converged = np.array([result[1] for result in results])
            iterations = np.array([result[2] for result in results])
            failed = failures(hz, logical_test, errors, guesses)
            if name == "random_serial":
                random_failed = failed
            rescued = (random_failed & ~failed).sum() if random_failed is not None else ""
            lost = (failed & ~random_failed).sum() if random_failed is not None else ""
            standard_error = np.sqrt(failed.mean() * (1 - failed.mean()) / settings.shots)
            seconds = time.time() - start_time
            writer.writerow([settings.code, error_rate, name, settings.shots, settings.max_iterations,
                             failed.mean(), standard_error, (~converged).sum(), (failed & converged).sum(),
                             iterations[converged].mean(), rescued, lost, seconds])
            file.flush()
            print(f"[[{settings.code}]] p={error_rate:.3f} {name:40s} LER={failed.mean():.4f} ± {standard_error:.4f}  "
                  f"stuck={(~converged).sum():4d}  wrong={(failed & converged).sum():4d}  "
                  f"iters={iterations[converged].mean():.2f}  vs random: +{rescued} -{lost}  ({seconds:.0f}s)",
                  flush=True)