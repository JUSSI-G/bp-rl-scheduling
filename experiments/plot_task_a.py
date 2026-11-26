import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

SCHEDULES = ["flooding", "serial", "random_serial"]


def main():
    parser = argparse.ArgumentParser(description="Figure for Task A.")
    parser.add_argument("--input", default="results/task_a.csv")
    parser.add_argument("--output", default="figures/task_a.png")
    parser.add_argument("--show", action="store_true")
    settings = parser.parse_args()

    with open(settings.input) as file:
        rows = list(csv.DictReader(file))
    code_names = sorted({row["code"] for row in rows}, key=lambda name: int(name.split("_")[0]))

    figure, axes = plt.subplots(2, len(code_names), figsize=(6 * len(code_names), 9))

    for column, code_name in enumerate(code_names):
        for schedule in SCHEDULES:
            selected = [
                row for row in rows if row["code"] == code_name and row["schedule"] == schedule
            ]
            error_rates = np.array([float(row["error_rate"]) for row in selected])
            lers = np.array([float(row["ler"]) for row in selected])
            shots = np.array([int(row["shots"]) for row in selected])
            standard_errors = np.sqrt(lers * (1 - lers) / shots)
            mean_iterations = np.array([float(row["mean_iterations"]) for row in selected])

            has_failures = lers > 0
            axes[0, column].errorbar(
                error_rates[has_failures],
                lers[has_failures],
                yerr=standard_errors[has_failures],
                marker="o",
                capsize=3,
                label=schedule,
            )
            axes[1, column].plot(error_rates, mean_iterations, marker="o", label=schedule)

        axes[0, column].set_xscale("log")
        axes[0, column].set_yscale("log")
        axes[0, column].set_title(f"[[{code_name.replace('_', ',')}]]: logical error rate")
        axes[0, column].set_xlabel("physical error rate p")
        axes[0, column].set_ylabel("logical error rate")
        axes[0, column].grid(True, which="both", alpha=0.3)
        axes[0, column].legend()

        axes[1, column].set_title(f"[[{code_name.replace('_', ',')}]]: iterations to converge")
        axes[1, column].set_xlabel("physical error rate p")
        axes[1, column].set_ylabel("mean iterations (converged shots)")
        axes[1, column].grid(True, alpha=0.3)
        axes[1, column].legend()

    figure.tight_layout()
    figure.savefig(settings.output, dpi=150)
    if settings.show:
        plt.show()


if __name__ == "__main__":
    main()
