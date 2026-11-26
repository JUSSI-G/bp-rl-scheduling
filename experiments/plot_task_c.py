import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Figure for Task C.")
    parser.add_argument("--check-input", default="results/task_c_long.csv")
    parser.add_argument("--qubit-input", default="results/task_c_rls.csv")
    parser.add_argument("--output", default="figures/task_c.png")
    parser.add_argument("--show", action="store_true")
    settings = parser.parse_args()

    with open(settings.check_input) as file:
        check_rows = {row["schedule"]: row for row in csv.DictReader(file)}
    with open(settings.qubit_input) as file:
        qubit_rows = {row["schedule"]: row for row in csv.DictReader(file)}

    entries = [
        ("flooding", check_rows["flooding"], "tab:blue"),
        ("check serial", check_rows["serial"], "tab:orange"),
        ("check random", check_rows["random_serial"], "tab:green"),
        ("own RL agent\n(check order)", check_rows["learned"], "tab:red"),
        ("qubit random", qubit_rows["qubit_random_serial"], "tab:olive"),
        ("RL-S\n(qubit order)", qubit_rows["rl_s"], "tab:purple"),
    ]
    labels = [label for label, row, color in entries]
    colors = [color for label, row, color in entries]
    lers = np.array([float(row["ler"]) for label, row, color in entries])
    standard_errors = np.array([float(row["standard_error"]) for label, row, color in entries])
    shots = np.array([int(row["shots"]) for label, row, color in entries])
    non_converged = np.array([int(row["non_converged"]) for label, row, color in entries]) / shots
    wrong = np.array([int(row["wrong"]) for label, row, color in entries]) / shots
    iterations = np.array([float(row["mean_iterations"]) for label, row, color in entries])
    positions = np.arange(len(entries))

    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(16, 4.8))

    left.bar(positions, lers, yerr=standard_errors, capsize=4, color=colors)
    left.set_ylim(0.14, 0.185)
    left.set_title("Logical error rate")
    left.set_ylabel("logical error rate")

    middle.bar(positions, non_converged, color="tab:gray", label="BP stuck (not converged)")
    middle.bar(
        positions,
        wrong,
        bottom=non_converged,
        color="black",
        alpha=0.75,
        label="converged but wrong",
    )
    middle.set_title("How decoding fails")
    middle.set_ylabel("fraction of shots")
    middle.set_ylim(0, 0.215)
    middle.legend(loc="upper right", fontsize=9)

    right.bar(positions, iterations, color=colors)
    right.set_title("Iterations to converge")
    right.set_ylabel("mean iterations (converged shots)")

    for axis in (left, middle, right):
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        axis.grid(True, axis="y", alpha=0.3)

    figure.suptitle(
        "[[72,12,6]], p = 0.05, 10 000 shots: learned schedules change how BP fails, not how often"
    )
    figure.tight_layout()
    figure.savefig(settings.output, dpi=150)
    if settings.show:
        plt.show()


if __name__ == "__main__":
    main()
