import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("144", 0.05, "results/night2_144.csv", "[[144,12,12]], p = 0.05, 200 iterations"),
    ("288", 0.06, "results/night2_288_iter200.csv", "[[288,12,18]], p = 0.06, 200 iterations"),
]
AGENTS = [
    ("results/learned_weights.json", "72-only, 3k\n(s123)", "control"),
    ("results/learned_weights_mixture_seed123.json", "mixture\ns123", "mixture"),
    ("results/learned_weights_mixture_seed1.json", "mixture\ns1", "mixture"),
    ("results/learned_weights_mixture_seed2.json", "mixture\ns2", "mixture"),
    ("results/learned_weights_long.json", "72-only, 15k\n(headline)", "headline"),
]
LIGHT_THEME = {
    "style": {},
    "colors": {"control": "tab:blue", "mixture": "tab:purple", "headline": "tab:gray"},
    "zero_line": "black",
}
DARK_THEME = {
    "style": {
        "figure.facecolor": "#1b1c20",
        "axes.facecolor": "#1b1c20",
        "savefig.facecolor": "#1b1c20",
        "axes.edgecolor": "#d0d0d0",
        "axes.labelcolor": "#e6e6e6",
        "text.color": "#e6e6e6",
        "xtick.color": "#d0d0d0",
        "ytick.color": "#d0d0d0",
        "grid.color": "#5a5a5a",
        "legend.facecolor": "#1b1c20",
        "legend.edgecolor": "#5a5a5a",
    },
    "colors": {"control": "#4c9be8", "mixture": "#b07fe0", "headline": "#9a9a9a"},
    "zero_line": "#e6e6e6",
}


def find_row(rows, code, error_rate, schedule):
    """Row of the CSV with the given code, error rate and schedule name."""
    for row in rows:
        if (
            row["code"] == code
            and abs(float(row["error_rate"]) - error_rate) < 1e-9
            and row["schedule"] == schedule
        ):
            return row


def reduction_vs_random(agent_row, random_row):
    """Fewer failures than random serial in %, and its paired standard error."""
    shots = int(agent_row["shots"])
    rescued = int(agent_row["rescued_vs_random"])
    lost = int(agent_row["lost_vs_random"])
    random_ler = float(random_row["ler"])
    difference = (rescued - lost) / shots
    difference_error = np.sqrt(rescued + lost - (rescued - lost) ** 2 / shots) / shots
    return 100 * difference / random_ler, 100 * difference_error / random_ler


def load_panels(task_d_file):
    """Reductions and errors of every agent, for every panel."""
    with open(task_d_file) as file:
        task_d_rows = list(csv.DictReader(file))
    panels = []
    for code, error_rate, headline_file, title in PANELS:
        with open(headline_file) as file:
            headline_rows = list(csv.DictReader(file))
        random_row = find_row(task_d_rows, code, error_rate, "random_serial")
        headline_random_row = find_row(headline_rows, code, error_rate, "random_serial")
        assert random_row["ler"] == headline_random_row["ler"]
        values = []
        for weights_file, label, group in AGENTS:
            rows = headline_rows if group == "headline" else task_d_rows
            agent_row = find_row(rows, code, error_rate, weights_file)
            values.append(reduction_vs_random(agent_row, random_row))
        panels.append((title, values))
    return panels


def draw(panels, theme, output):
    """Draw both panels with the given theme and save the figure."""
    with plt.rc_context(theme["style"]):
        figure, axes = plt.subplots(1, len(panels), figsize=(12.8, 7.2), sharey=True)
        positions = np.arange(len(AGENTS))
        for axis, (title, values) in zip(axes, panels):
            axis.axhline(0, color=theme["zero_line"], linestyle="--", linewidth=1.2)
            axis.text(len(AGENTS) - 0.5, 0.4, "random serial", ha="right", va="bottom", fontsize=10)
            for position, (value, error), (weights_file, label, group) in zip(
                positions, values, AGENTS
            ):
                color = theme["colors"][group]
                axis.errorbar(
                    position, value, yerr=error, fmt="o", markersize=10, capsize=5, color=color
                )
                axis.text(
                    position + 0.12,
                    value,
                    f"{value:.1f}%",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=color,
                )
            axis.set_xticks(positions)
            axis.set_xticklabels([label for weights_file, label, group in AGENTS], fontsize=10)
            axis.set_xlim(-0.5, len(AGENTS) - 0.3)
            axis.set_title(title, fontsize=13)
            axis.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("fewer failures than random serial (%)")
        axes[0].set_ylim(-4, 31)
        figure.suptitle("Task D: training on a mixture of Tanner graphs", fontsize=15)
        figure.tight_layout()
        figure.savefig(output, dpi=200)
    return figure


def main():
    parser = argparse.ArgumentParser(description="Figure for Task D (mixture of Tanner graphs).")
    parser.add_argument("--input", default="results/task_d_mixture.csv")
    parser.add_argument("--output", default="figures/task_d_mixture.png")
    parser.add_argument("--dark-output", default="figures/task_d_mixture_dark.png")
    parser.add_argument("--show", action="store_true")
    settings = parser.parse_args()

    panels = load_panels(settings.input)
    for title, values in panels:
        print(title, [f"{value:.1f} +- {error:.1f}" for value, error in values])
    draw(panels, LIGHT_THEME, settings.output)
    draw(panels, DARK_THEME, settings.dark_output)
    if settings.show:
        plt.show()


if __name__ == "__main__":
    main()
