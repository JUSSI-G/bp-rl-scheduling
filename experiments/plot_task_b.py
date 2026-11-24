import numpy as np
import matplotlib.pyplot as plt

code_name = "72_12_6"
data = np.load(f"results/task_b_{code_name}.npz")
random_failed = data["random_failed"]
number_of_shots, number_of_seeds = random_failed.shape
seed_lers = random_failed.mean(axis=0)

seed_counts = np.arange(1, number_of_seeds + 1)
oracle_lers = [random_failed[:, :count].all(axis=1).mean() for count in seed_counts]

figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

left.hist(seed_lers, bins=10, color="tab:green", alpha=0.7, label="random serial (one bar per seed)")
left.axvline(data["flooding_failed"].mean(), color="tab:blue", linewidth=2, label="flooding")
left.axvline(data["serial_failed"].mean(), color="tab:orange", linewidth=2, label="serial (fixed order)")
left.set_xlabel("logical error rate")
left.set_ylabel("number of seeds")
left.set_title(f"[[{code_name.replace('_', ',')}]], p={float(data['error_rate'])}: LER per random seed")
left.legend()

right.plot(seed_counts, oracle_lers, marker="o", color="tab:green", label="best of K random orders (oracle)")
right.axhline(data["serial_failed"].mean(), color="tab:orange", linestyle="--", label="serial (fixed order)")
right.axhline(seed_lers.mean(), color="tab:green", linestyle=":", label="random serial, single seed (mean)")
right.set_xlabel("number of random orders tried (K)")
right.set_ylabel("logical error rate")
right.set_title("How much could a better order help?")
right.set_ylim(bottom=0)
right.legend()

figure.tight_layout()
figure.savefig(f"figures/task_b_{code_name}.png", dpi=150)
plt.show()