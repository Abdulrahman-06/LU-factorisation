# gaussian_elimination.py
# Notes-style Gaussian elimination (forward elimination only), made self-contained.
# Modifies A and b in place. No pivoting, to match the lab’s constraints.

import numpy as np

# ---- helpers expected by the notes code ----

def print_array(X):
    """Lightweight pretty-printer used by verbose mode."""
    # handle vectors cleanly whether shaped (n,) or (n,1)
    if X.ndim == 1 or (X.ndim == 2 and 1 in X.shape):
        arr = np.asarray(X).reshape(-1)
        print("[ " + "  ".join(f"{v: .6g}" for v in arr) + " ]")
    else:
        for row in np.asarray(X):
            print("[ " + "  ".join(f"{v: .6g}" for v in row) + " ]")

def system_size(A, b):
    """Return n for an n×n system Ax = b; checks b's first dim matches A."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    n, _ = A.shape
    if b.shape[0] != n:
        raise ValueError(f"Incompatible dimensions: A is {A.shape}, b is {b.shape}")
    return n

def row_add(A, b, j, alpha, i):
    """R_j ← R_j + alpha * R_i  (apply to both A and b)."""
    A[j, :] += alpha * A[i, :]
    if b.ndim == 1:
        b[j] += alpha * b[i]
    else:
        b[j, :] += alpha * b[i, :]

def row_swap(A, b, i, j):
    """Swap rows i and j in A and b. (Not used here, but handy if needed.)"""
    A[[i, j], :] = A[[j, i], :]
    if b.ndim == 1:
        b[[i, j]] = b[[j, i]]
    else:
        b[[i, j], :] = b[[j, i], :]

# ---- main routine from the notes ----

def gaussian_elimination(A, b, verbose=False):
    """
    Perform Gaussian elimination (forward elimination) to reduce Ax=b
    to upper-triangular form. Works IN PLACE on A and b. No pivoting.
    """
    n = system_size(A, b)

    for i in range(n - 1):
        if verbose:
            print(f"eliminating column {i}")
        # eliminate entries below the diagonal in column i
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            if verbose:
                print(f"  R{j} ← R{j} - ({factor}) * R{i}")
            row_add(A, b, j, -factor, i)

        if verbose:
            print("\nnew system (A):")
            print_array(A)
            print("new RHS (b):")
            print_array(b)
            print()

if __name__ == "__main__":
    # quick smoke test
    A = np.array([[2.0, 1.0, -1.0],
                  [-3.0, -1.0,  2.0],
                  [-2.0,  1.0,  2.0]])
    b = np.array([[8.0], [-11.0], [-3.0]])
    Ac, bc = A.copy(), b.copy()
    gaussian_elimination(Ac, bc, verbose=True)
    print("Forward elimination completed.")
