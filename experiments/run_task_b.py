import argparse
import time

import numpy as np

from src.bp import BeliefPropagation
from src.codes import build_code
from src.sim import failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(description="Task B: how much does the serial order matter?")
    parser.add_argument("--code", default="72")
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="results/task_b_72_12_6.npz")
    settings = parser.parse_args()

    hx, hz, logical_test = build_code(settings.code)
    number_of_qubits = hx.shape[1]

    error_rng = np.random.default_rng(settings.seed)
    errors = sample_errors(number_of_qubits, settings.error_rate, settings.shots, error_rng)
    all_syndromes = syndromes(hz, errors)
    decoder = BeliefPropagation(hz, settings.error_rate, settings.max_iterations)

    def decode_all(schedule, seed):
        guesses = []
        for shot, syndrome in enumerate(all_syndromes):
            shot_rng = np.random.default_rng([seed, shot])
            guess = decoder.decode(syndrome, schedule, shot_rng)[0]
            guesses.append(guess)
        return failures(hz, logical_test, errors, np.array(guesses, dtype=np.uint8))

    start_time = time.time()
    flooding_failed = decode_all("flooding", 0)
    serial_failed = decode_all("serial", 0)
    random_failed = np.zeros((settings.shots, settings.seeds), dtype=bool)
    for seed in range(settings.seeds):
        random_failed[:, seed] = decode_all("random_serial", seed)
        print(f"seed {seed:2d}: LER={random_failed[:, seed].mean():.4f}")

    np.savez(
        settings.output,
        flooding_failed=flooding_failed,
        serial_failed=serial_failed,
        random_failed=random_failed,
        error_rate=settings.error_rate,
        error_seed=settings.seed,
        max_iterations=settings.max_iterations,
    )

    seed_lers = random_failed.mean(axis=0)
    always_fail = random_failed.all(axis=1)
    never_fail = (~random_failed).all(axis=1)
    sometimes_fail = ~always_fail & ~never_fail

    print()
    print(f"flooding LER         {flooding_failed.mean():.4f}")
    print(f"serial LER           {serial_failed.mean():.4f}")
    print(
        f"random serial LER    mean {seed_lers.mean():.4f}  std {seed_lers.std():.4f}  "
        f"min {seed_lers.min():.4f}  max {seed_lers.max():.4f}"
    )
    print(f"oracle LER (any of {settings.seeds} seeds succeeds)  {always_fail.mean():.4f}")
    print(
        f"shots: always succeed {never_fail.sum()}, always fail {always_fail.sum()}, "
        f"depends on order {sometimes_fail.sum()}"
    )
    print(f"time {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
