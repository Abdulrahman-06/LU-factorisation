import numpy as np
def lu_factorisation(A):
    
    
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    L, U = np.zeros_like(A), np.zeros_like(A)

    for k in range(n):
        for j in range(k, n):
            total = 0
            for s in range(k):
                total += L[k, s] * U[s, j]
            U[k, j] = A[k, j] - total

        L[k, k] = 1

        for i in range(k + 1, n):
            total = 0
            for s in range(k):
                total += L[i, s] * U[s, k]

            if U[k, k] == 0:
                raise ZeroDivisionError("Zero pivot encountered")

            L[i, k] = (A[i, k] - total) / U[k, k]

    return L, U