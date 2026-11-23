import time
import numpy as np
from ldpc import BpDecoder
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures
from src.bp import BeliefPropagation

hx, hz = bb_code(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
logical_test = gf2_kernel(hx)
error_rate = 0.05
number_of_shots = 2000

error_rng = np.random.default_rng(1)
errors = sample_errors(72, error_rate, number_of_shots, error_rng)
all_syndromes = syndromes(hz, errors)

library_decoder = BpDecoder(hz, error_rate=error_rate, max_iter=50,
                            bp_method="product_sum", schedule="parallel")
library_guesses = np.array([library_decoder.decode(s) for s in all_syndromes], dtype=np.uint8)
library_ler = failures(hz, logical_test, errors, library_guesses).mean()
print(f"ldpc flooding    LER={library_ler:.4f}")

for schedule in ["flooding", "serial", "random_serial"]:
    decoder = BeliefPropagation(hz, error_rate)
    schedule_rng = np.random.default_rng(7)
    start_time = time.time()
    results = [decoder.decode(s, schedule, schedule_rng) for s in all_syndromes]
    guesses = np.array([guess for guess, converged, iterations in results], dtype=np.uint8)
    converged = np.array([converged for guess, converged, iterations in results])
    iterations = np.array([iterations for guess, converged, iterations in results])
    failed = failures(hz, logical_test, errors, guesses)
    print(f"{schedule:15s}  LER={failed.mean():.4f}  "
          f"non-converged={(~converged).sum()}  wrong={(failed & converged).sum()}  "
          f"mean iterations={iterations[converged].mean():.2f}  "
          f"time={time.time() - start_time:.1f}s")