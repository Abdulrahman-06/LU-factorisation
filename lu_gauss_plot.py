import time
import numpy as np
import matplotlib.pyplot as plt

from lu_factorisation import lu_factorisation
from gaussian_elimination import gaussian_elimination

rng = np.random.default_rng(42)

def best_time(run, repeats=3):
    """Return the best runtime (seconds) over several repeats."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best

sizes = [50, 100, 150, 200, 250]
lu_times = []
ge_times = []

for n in sizes:
    A = rng.standard_normal((n, n))
    b = rng.standard_normal((n, 1))

    def do_lu():
        lu_factorisation(A)

    lu_times.append(best_time(do_lu))

    def do_ge():
        Ac = A.copy()
        bc = b.copy()
        gaussian_elimination(Ac, bc, verbose=False)

    ge_times.append(best_time(do_ge))

plt.figure()
plt.plot(sizes, lu_times, marker="o", label="LU factorisation (my code)")
plt.plot(sizes, ge_times, marker="s", label="Gaussian elimination (notes)")
plt.xlabel("Matrix size n")
plt.ylabel("Best runtime (s)")
plt.title("Runtime comparison: LU vs Gaussian elimination")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


