import numpy as np
from lu_factorisation import lu_factorisation

def generate_safe_system(n):
    """Generate a random invertible system Ax = b with known solution."""
    A = np.random.rand(n, n) * 10
    while np.linalg.matrix_rank(A) < n:
        A = np.random.rand(n, n) * 10
    x = np.random.rand(n)
    b = A @ x
    return A, b, x


def determinant(A):
    """Compute determinant using LU factorisation."""
    n = A.shape[0]
    L, U = lu_factorisation(A)
    det_L = np.prod(np.diag(L))
    det_U = np.prod(np.diag(U))
    return det_L * det_U


if __name__ == "__main__":
    A_large, b_large, x_large = generate_safe_system(100)
    det_A_large = determinant(A_large)
    print("Determinant of A_large:", det_A_large)
