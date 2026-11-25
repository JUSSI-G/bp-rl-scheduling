import json
import time
import numpy as np
from src.codes import bb_code
from src.sim import sample_errors, syndromes
from src.vertical_bp import VerticalBeliefPropagation

l, m = 6, 6
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
training_error_rates = [0.03, 0.04, 0.05, 0.06, 0.07]
training_episodes = 20000
max_iterations = 20
learning_rate = 0.1
discount = 0.9
start_exploration = 1.0
final_exploration = 0.05
training_seed = 123

hx, hz = bb_code(l, m, a_terms, b_terms)
number_of_qubits = hx.shape[1]
checks_per_qubit = hz.sum(axis=0)[0]
number_of_states = 2 ** checks_per_qubit
q_table = np.zeros((number_of_states, number_of_qubits))
rng = np.random.default_rng(training_seed)

start_time = time.time()
for episode in range(1, training_episodes + 1):
    exploration = max(final_exploration,
                      start_exploration * (1 - (episode - 1) / (training_episodes - 1)))
    error_rate = rng.choice(training_error_rates)
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

            q_table[state, qubit] += learning_rate * (reward + discount * future_value - q_table[state, qubit])
            if finished:
                break

    if episode % 2000 == 0:
        print(f"episode {episode}: exploration={exploration:.2f}  "
              f"mean Q per state={np.round(q_table.mean(axis=1), 2)}  ({time.time() - start_time:.0f}s)")

np.save("results/rls_q_table.npy", q_table)
with open("results/rls_settings.json", "w") as file:
    json.dump({"training_error_rates": training_error_rates, "training_episodes": training_episodes,
               "max_iterations": max_iterations, "learning_rate": learning_rate, "discount": discount,
               "start_exploration": start_exploration, "final_exploration": final_exploration,
               "training_seed": training_seed}, file, indent=2)