import argparse
import json
import time

import numpy as np

from src.codes import build_code
from src.sim import sample_errors, syndromes
from src.vertical_bp import VerticalBeliefPropagation


def main():
    parser = argparse.ArgumentParser(description="Train the RL-S tabular Q-learning qubit schedule.")
    parser.add_argument("--code", default="72")
    parser.add_argument(
        "--error-rates", type=float, nargs="+", default=[0.03, 0.04, 0.05, 0.06, 0.07]
    )
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--start-exploration", type=float, default=1.0)
    parser.add_argument("--final-exploration", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/rls_q_table.npy")
    parser.add_argument("--settings-output", default="results/rls_settings.json")
    settings = parser.parse_args()

    episodes = settings.episodes
    max_iterations = settings.max_iterations
    learning_rate = settings.learning_rate
    discount = settings.discount

    hx, hz, _ = build_code(settings.code)
    number_of_qubits = hx.shape[1]
    checks_per_qubit = hz.sum(axis=0)[0]
    number_of_states = 2**checks_per_qubit
    q_table = np.zeros((number_of_states, number_of_qubits))
    rng = np.random.default_rng(settings.seed)

    start_time = time.time()
    for episode in range(1, episodes + 1):
        exploration = max(
            settings.final_exploration,
            settings.start_exploration * (1 - (episode - 1) / (episodes - 1)),
        )
        error_rate = rng.choice(settings.error_rates)
        decoder = VerticalBeliefPropagation(hz, error_rate, max_iterations)
        error = sample_errors(number_of_qubits, error_rate, 1, rng)
        decoder.start(syndromes(hz, error)[0])

        finished = not decoder.unsatisfied.any()
        for iteration in range(max_iterations):
            if finished:
                break
            remaining = np.ones(number_of_qubits, dtype=bool)
            for step in range(number_of_qubits):
                candidates = np.flatnonzero(remaining)
                states = decoder.qubit_states()
                if rng.random() < exploration:
                    qubit = rng.choice(candidates)
                else:
                    qubit = candidates[np.argmax(q_table[states[candidates], candidates])]
                state = states[qubit]

                unsatisfied_before = decoder.unsatisfied.sum()
                decoder.update_qubit(qubit)
                unsatisfied_after = decoder.unsatisfied.sum()
                remaining[qubit] = False

                reward = (unsatisfied_before - unsatisfied_after) / checks_per_qubit
                finished = unsatisfied_after == 0
                if finished:
                    reward += 1.0
                    future_value = 0.0
                else:
                    next_candidates = np.flatnonzero(remaining)
                    if len(next_candidates) == 0:
                        future_value = 0.0
                    else:
                        next_states = decoder.qubit_states()[next_candidates]
                        future_value = q_table[next_states, next_candidates].max()

                q_table[state, qubit] += learning_rate * (
                    reward + discount * future_value - q_table[state, qubit]
                )
                if finished:
                    break

        if episode % 2000 == 0:
            print(
                f"episode {episode}: exploration={exploration:.2f}  "
                f"mean Q per state={np.round(q_table.mean(axis=1), 2)}  "
                f"({time.time() - start_time:.0f}s)"
            )

    np.save(settings.output, q_table)
    with open(settings.settings_output, "w") as file:
        json.dump(
            {
                "training_error_rates": settings.error_rates,
                "training_episodes": episodes,
                "max_iterations": max_iterations,
                "learning_rate": learning_rate,
                "discount": discount,
                "start_exploration": settings.start_exploration,
                "final_exploration": settings.final_exploration,
                "training_seed": settings.seed,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
