import csv
import numpy as np
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/night2_144.csv")))
random_row = [row for row in rows if row["schedule"] == "random_serial"][0]
entries = [
    ("results/learned_weights_long.json", "original\nseed 123", "tab:blue"),
    ("results/weights_72_original_s1.json", "original\nseed 1", "tab:blue"),
    ("results/weights_72_original_s2.json", "original\nseed 2", "tab:blue"),
    ("results/weights_72_original_s3.json", "original\nseed 3", "tab:blue"),
    ("results/weights_72_original_60k.json", "original\n60k shots", "tab:blue"),
    ("results/rule_residual_first.json", "rule:\nresidual first", "tab:gray"),
    ("results/weights_72_residual_s123.json", "+residual\nseed 123", "tab:orange"),
    ("results/weights_72_residual_s1.json", "+residual\nseed 1", "tab:orange"),
    ("results/weights_72_residual_s2.json", "+residual\nseed 2", "tab:orange"),
]
by_schedule = {row["schedule"]: row for row in rows}
lers = np.array([float(by_schedule[name]["ler"]) for name, label, color in entries])
errors = np.array([float(by_schedule[name]["standard_error"]) for name, label, color in entries])
random_ler = float(random_row["ler"])
random_error = float(random_row["standard_error"])
positions = np.arange(len(entries))

figure, axis = plt.subplots(figsize=(14, 4.4))
axis.axhspan(random_ler - random_error, random_ler + random_error, color="tab:green", alpha=0.2)
axis.axhline(random_ler, color="tab:green", linewidth=2, label=f"random serial ({random_ler:.4f})")
for position, ler, error, (name, label, color) in zip(positions, lers, errors, entries):
    axis.errorbar(position, ler, yerr=error, fmt="o", markersize=9, capsize=4, color=color)
axis.set_xticks(positions)
axis.set_xticklabels([label for name, label, color in entries], fontsize=9)
axis.set_ylabel("logical error rate")
axis.set_title("[[144,12,12]], p = 0.05, 200 iterations, 20 000 shots: independent training runs on [[72,12,6]]")
axis.grid(True, axis="y", alpha=0.3)
axis.legend(loc="upper left")
figure.tight_layout()
figure.savefig("figures/robustness.png", dpi=150)
plt.show()