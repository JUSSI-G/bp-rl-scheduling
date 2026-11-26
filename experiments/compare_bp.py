import argparse
import time

import numpy as np
from ldpc import BpDecoder

from src.bp import BeliefPropagation
from src.codes import build_code
from src.sim import failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(description="Check the own BP against ldpc's BpDecoder.")
    parser.add_argument("--code", default="72")
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--schedule-seed", type=int, default=7)
    settings = parser.parse_args()

    error_rate = settings.error_rate
    hx, hz, logical_test = build_code(settings.code)
    error_rng = np.random.default_rng(settings.seed)
    errors = sample_errors(hx.shape[1], error_rate, settings.shots, error_rng)
    all_syndromes = syndromes(hz, errors)

    library_decoder = BpDecoder(
        hz,
        error_rate=error_rate,
        max_iter=settings.max_iterations,
        bp_method="product_sum",
        schedule="parallel",
    )
    library_guesses = np.array([library_decoder.decode(s) for s in all_syndromes], dtype=np.uint8)
    library_ler = failures(hz, logical_test, errors, library_guesses).mean()
    print(f"ldpc flooding    LER={library_ler:.4f}")

    for schedule in ["flooding", "serial", "random_serial"]:
        decoder = BeliefPropagation(hz, error_rate, settings.max_iterations)
        schedule_rng = np.random.default_rng(settings.schedule_seed)
        start_time = time.time()
        results = [decoder.decode(s, schedule, schedule_rng) for s in all_syndromes]
        guesses = np.array([result[0] for result in results], dtype=np.uint8)
        converged = np.array([result[1] for result in results])
        iterations = np.array([result[2] for result in results])
        failed = failures(hz, logical_test, errors, guesses)
        print(
            f"{schedule:15s}  LER={failed.mean():.4f}  "
            f"non-converged={(~converged).sum()}  wrong={(failed & converged).sum()}  "
            f"mean iterations={iterations[converged].mean():.2f}  "
            f"time={time.time() - start_time:.1f}s"
        )


if __name__ == "__main__":
    main()
