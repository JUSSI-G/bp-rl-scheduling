import csv
import numpy as np
import matplotlib.pyplot as plt

agent = "results/learned_weights_long.json"
settings = [
    ("results/overnight_72.csv", 0.05, "[[72,12,6]]\np=0.05\n(trained here)"),
    ("results/overnight_144_iter50.csv", 0.04, "[[144,12,12]]\np=0.04"),
    ("results/overnight_144_iter50.csv", 0.05, "[[144,12,12]]\np=0.05"),
    ("results/overnight_144_iter50.csv", 0.06, "[[144,12,12]]\np=0.06"),
    ("results/night2_288.csv", 0.05, "[[288,12,18]]\np=0.05"),
    ("results/night2_288.csv", 0.06, "[[288,12,18]]\np=0.06"),
    ("results/night2_288_iter200.csv", 0.06, "[[288,12,18]]\np=0.06\n200 iterations"),
]


def find_row(rows, error_rate, schedule):
    for row in rows:
        if abs(float(row["error_rate"]) - error_rate) < 1e-9 and row["schedule"] == schedule:
            return row


labels = []
reductions = []
uncertainties = []
for file_name, error_rate, label in settings:
    rows = list(csv.DictReader(open(file_name)))
    random_row = find_row(rows, error_rate, "random_serial")
    agent_row = find_row(rows, error_rate, agent)
    shots = int(agent_row["shots"])
    rescued = int(agent_row["rescued_vs_random"])
    lost = int(agent_row["lost_vs_random"])
    random_ler = float(random_row["ler"])
    difference = (rescued - lost) / shots
    difference_error = np.sqrt(rescued + lost - (rescued - lost) ** 2 / shots) / shots
    labels.append(label)
    reductions.append(100 * difference / random_ler)
    uncertainties.append(100 * difference_error / random_ler)

colors = ["tab:gray"] + ["tab:blue"] * 3 + ["tab:purple"] * 2 + ["darkviolet"]
positions = np.arange(len(labels))
figure, axis = plt.subplots(figsize=(14, 4.4))
axis.bar(positions, reductions, yerr=uncertainties, capsize=4, color=colors)
axis.axhline(0, color="black", linewidth=0.8)
axis.set_ylim(-8, 32)
for position, value, uncertainty in zip(positions, reductions, uncertainties):
    axis.text(position, value + uncertainty + 0.8, f"{value:.0f}%", ha="center", fontsize=11, fontweight="bold")
axis.set_xticks(positions)
axis.set_xticklabels(labels, fontsize=9)
axis.set_ylabel("fewer failures than random serial (%)")
axis.set_title("Rule learned on [[72,12,6]], applied unchanged to larger codes (50 iterations unless noted)")
axis.grid(True, axis="y", alpha=0.3)
figure.tight_layout()
figure.savefig("figures/transfer.png", dpi=150)
plt.show()