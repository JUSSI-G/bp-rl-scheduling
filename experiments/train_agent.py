import argparse
import json
import time

import numpy as np

from src.codes import BB_CODES, build_code
from src.rl import FEATURE_NAMES, LearnedScheduleDecoder
from src.sim import failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(description="Train the check-schedule policy with REINFORCE.")
    parser.add_argument("--code", default="72", choices=BB_CODES.keys())
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--iteration-penalty", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--output", required=True)
    settings = parser.parse_args()

    hx, hz, logical_test = build_code(settings.code)
    decoder = LearnedScheduleDecoder(
        hz, settings.error_rate, np.zeros(len(FEATURE_NAMES)), settings.max_iterations
    )
    residual_position = FEATURE_NAMES.index("residual")
    error_rng = np.random.default_rng(settings.seed)
    if settings.policy_seed is None:
        settings.policy_seed = settings.seed + 1
    policy_rng = np.random.default_rng(settings.policy_seed)

    start_time = time.time()
    report_every = max(1, settings.shots // 10)
    for shot in range(1, settings.shots + 1):
        error = sample_errors(hx.shape[1], settings.error_rate, 1, error_rng)
        syndrome = syndromes(hz, error)[0]

        rewards = []
        gradients = []
        for _ in range(settings.rollouts):
            guess, converged, iterations, gradient = decoder.decode(
                syndrome, policy_rng, explore=True
            )
            succeeded = not failures(hz, logical_test, error, guess[None])[0]
            rewards.append(float(succeeded) - settings.iteration_penalty * iterations)
            gradients.append(gradient / iterations)

        rewards = np.array(rewards)
        advantages = rewards - rewards.mean()
        for advantage, gradient in zip(advantages, gradients):
            decoder.weights += settings.learning_rate * advantage * gradient
        if settings.no_residual:
            decoder.weights[residual_position] = 0.0

        if shot % report_every == 0:
            weights_text = ", ".join(
                f"{name}={weight:+.2f}" for name, weight in zip(FEATURE_NAMES, decoder.weights)
            )
            print(f"shot {shot}: {weights_text}  ({time.time() - start_time:.0f}s)", flush=True)

    with open(settings.output, "w") as file:
        json.dump(
            {"feature_names": FEATURE_NAMES, "weights": decoder.weights.tolist(), **vars(settings)},
            file,
            indent=2,
        )
    print(f"saved {settings.output}")


if __name__ == "__main__":
    main()
