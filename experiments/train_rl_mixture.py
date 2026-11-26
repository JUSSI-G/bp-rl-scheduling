import argparse
import json
import time

import numpy as np

from src.codes import BB_CODES, build_code
from src.rl import FEATURE_NAMES, LearnedScheduleDecoder
from src.sim import failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(
        description="Task D: train one check-schedule policy on a mixture of Tanner graphs."
    )
    parser.add_argument("--codes", nargs="+", default=["72", "144"], choices=BB_CODES.keys())
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--iteration-penalty", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--output", default=None)
    settings = parser.parse_args()
    if settings.output is None:
        settings.output = f"results/learned_weights_mixture_seed{settings.seed}.json"

    weights = np.zeros(len(FEATURE_NAMES))
    codes = {}
    for code in settings.codes:
        hx, hz, logical_test = build_code(code)
        decoder = LearnedScheduleDecoder(hz, settings.error_rate, weights, settings.max_iterations)
        decoder.weights = weights
        codes[code] = (hx, hz, logical_test, decoder)
    residual_position = FEATURE_NAMES.index("residual")
    code_rng, error_rng, policy_rng = [
        np.random.default_rng(child) for child in np.random.SeedSequence(settings.seed).spawn(3)
    ]

    shots_per_code = {code: 0 for code in settings.codes}
    failing_rollouts_per_code = {code: 0 for code in settings.codes}
    mixed_outcome_shots_per_code = {code: 0 for code in settings.codes}
    start_time = time.time()
    report_every = max(1, settings.shots // 10)
    for shot in range(1, settings.shots + 1):
        code = settings.codes[code_rng.integers(len(settings.codes))]
        hx, hz, logical_test, decoder = codes[code]
        error = sample_errors(hx.shape[1], settings.error_rate, 1, error_rng)
        syndrome = syndromes(hz, error)[0]

        rewards = []
        gradients = []
        successes = []
        for _ in range(settings.rollouts):
            guess, converged, iterations, gradient = decoder.decode(
                syndrome, policy_rng, explore=True
            )
            succeeded = not failures(hz, logical_test, error, guess[None])[0]
            successes.append(succeeded)
            rewards.append(float(succeeded) - settings.iteration_penalty * iterations)
            gradients.append(gradient / iterations)

        rewards = np.array(rewards)
        advantages = rewards - rewards.mean()
        for advantage, gradient in zip(advantages, gradients):
            weights += settings.learning_rate * advantage * gradient
        if not settings.residual:
            weights[residual_position] = 0.0

        shots_per_code[code] += 1
        failing_rollouts_per_code[code] += settings.rollouts - sum(successes)
        mixed_outcome_shots_per_code[code] += 0 < sum(successes) < settings.rollouts

        if shot % report_every == 0:
            weights_text = ", ".join(
                f"{name}={weight:+.2f}" for name, weight in zip(FEATURE_NAMES, weights)
            )
            print(f"shot {shot}: {weights_text}  ({time.time() - start_time:.0f}s)", flush=True)

    for code in settings.codes:
        print(
            f"[[{code}]]: {shots_per_code[code]} shots, "
            f"{failing_rollouts_per_code[code]} failing rollouts, "
            f"{mixed_outcome_shots_per_code[code]} shots where rollouts disagree on success"
        )
    print("final weights: " + ", ".join(f"{n}={w:+.4f}" for n, w in zip(FEATURE_NAMES, weights)))

    with open(settings.output, "w") as file:
        json.dump(
            {
                "feature_names": FEATURE_NAMES,
                "weights": weights.tolist(),
                **vars(settings),
                "shots_per_code": shots_per_code,
                "failing_rollouts_per_code": failing_rollouts_per_code,
                "mixed_outcome_shots_per_code": mixed_outcome_shots_per_code,
                "seconds": time.time() - start_time,
            },
            file,
            indent=2,
        )
    print(f"saved {settings.output}")


if __name__ == "__main__":
    main()
