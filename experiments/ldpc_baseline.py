import argparse

import matplotlib.pyplot as plt
import numpy as np
from ldpc import BpDecoder

from src.codes import build_code
from src.sim import failures, sample_errors, syndromes


def main():
    parser = argparse.ArgumentParser(description="Baseline: ldpc flooding BP on [[72,12,6]].")
    parser.add_argument("--code", default="72")
    parser.add_argument(
        "--error-rates",
        type=float,
        nargs="+",
        default=[0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    )
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="figures/ldpc_baseline_72.png")
    parser.add_argument("--show", action="store_true")
    settings = parser.parse_args()

    hx, hz, logical_test = build_code(settings.code)
    number_of_qubits = hx.shape[1]
    rng = np.random.default_rng(settings.seed)

    lers = []
    for error_rate in settings.error_rates:
        errors = sample_errors(number_of_qubits, error_rate, settings.shots, rng)
        all_syndromes = syndromes(hz, errors)
        decoder = BpDecoder(
            hz,
            error_rate=error_rate,
            max_iter=settings.max_iterations,
            bp_method="product_sum",
            schedule="parallel",
        )
        guesses = np.array([decoder.decode(syndrome) for syndrome in all_syndromes], dtype=np.uint8)
        failed = failures(hz, logical_test, errors, guesses)
        not_converged = ((errors ^ guesses) @ hz.T % 2).any(axis=1)
        lers.append(failed.mean())
        print(f"p={error_rate:.2f}  LER={failed.mean():.4f}  ({failed.sum()} failures)")
        print(
            f"      non-converged: {not_converged.sum()}, "
            f"logical (converged but wrong): {(failed & ~not_converged).sum()}"
        )

    plt.loglog(settings.error_rates, lers, "o-", label="ldpc BP, flooding")
    plt.xlabel("physical error rate p")
    plt.ylabel("logical error rate")
    plt.title("[[72,12,6]] BB code, code-capacity X noise")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(settings.output, dpi=150)
    if settings.show:
        plt.show()


if __name__ == "__main__":
    main()
