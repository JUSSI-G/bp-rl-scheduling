import json
import time
import numpy as np
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.rl import LearnedScheduleDecoder, feature_names

l, m = 6, 6
a_terms = [(3, 0), (0, 1), (0, 2)]
b_terms = [(0, 3), (1, 0), (2, 0)]
error_rate = 0.05
training_shots = 15000
rollouts_per_shot = 4
learning_rate = 0.15
iteration_penalty = 0.05
max_iterations = 20
training_seed = 123
policy_seed = 0

hx, hz = bb_code(l, m, a_terms, b_terms)
logical_test = gf2_kernel(hx)
decoder = LearnedScheduleDecoder(hz, error_rate, np.zeros(len(feature_names)), max_iterations)
error_rng = np.random.default_rng(training_seed)
policy_rng = np.random.default_rng(policy_seed)

start_time = time.time()
for shot in range(1, training_shots + 1):
    error = sample_errors(hx.shape[1], error_rate, 1, error_rng)
    syndrome = syndromes(hz, error)[0]

    rewards = []
    gradients = []
    for rollout in range(rollouts_per_shot):
        guess, converged, iterations, gradient = decoder.decode(syndrome, policy_rng, explore=True)
        succeeded = not failures(hz, logical_test, error, guess[None])[0]
        rewards.append(float(succeeded) - iteration_penalty * iterations)
        gradients.append(gradient / iterations)

    rewards = np.array(rewards)
    advantages = rewards - rewards.mean()
    for advantage, gradient in zip(advantages, gradients):
        decoder.weights += learning_rate * advantage * gradient

    if shot % 500 == 0:
        weights_text = ", ".join(f"{name}={weight:+.2f}" for name, weight in zip(feature_names, decoder.weights))
        print(f"shot {shot}: {weights_text}  ({time.time() - start_time:.0f}s)")

with open("results/learned_weights_long.json", "w") as file:
    json.dump({"feature_names": feature_names, "weights": decoder.weights.tolist(),
               "error_rate": error_rate, "training_shots": training_shots,
               "rollouts_per_shot": rollouts_per_shot, "learning_rate": learning_rate,
               "iteration_penalty": iteration_penalty, "max_iterations": max_iterations,
               "training_seed": training_seed, "policy_seed": policy_seed}, file, indent=2)
