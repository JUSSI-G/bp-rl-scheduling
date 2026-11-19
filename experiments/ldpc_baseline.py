import numpy as np
import matplotlib.pyplot as plt
from ldpc import BpDecoder
from src.codes import bb_code, gf2_kernel
from src.sim import sample_errors, syndromes, failures

hx, hz = bb_code(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
n = hx.shape[1]
K = gf2_kernel(hx)

ps = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
shots = 2000
rng = np.random.default_rng(1)

lers = []
for p in ps:
    err = sample_errors(n, p, shots, rng)
    s = syndromes(hz, err)
    dec = BpDecoder(hz, error_rate=p, max_iter=50,
                    bp_method="product_sum", schedule="parallel")
    ehat = np.array([dec.decode(si) for si in s], dtype=np.uint8)
    fail = failures(hz, K, err, ehat)
    not_conv = ((err ^ ehat) @ hz.T % 2).any(axis=1)
    lers.append(fail.mean())
    print(f"p={p:.2f}  LER={fail.mean():.4f}  ({fail.sum()} failures)")
    print(f"      non-converged: {not_conv.sum()}, logical (converged but wrong): {(fail & ~not_conv).sum()}")


plt.loglog(ps, lers, "o-", label="ldpc BP, flooding")
plt.xlabel("physical error rate p")
plt.ylabel("logical error rate")
plt.title("[[72,12,6]] BB code, code-capacity X noise")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("figures/ldpc_baseline_72.png", dpi=150)
plt.show()