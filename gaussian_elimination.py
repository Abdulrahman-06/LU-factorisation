def gaussian_elimination(A, b, verbose=False):
    """
    Perform Gaussian elimination to reduce a linear system to upper triangular
    form.

    This function performs **forward elimination** to transform the coefficient
    matrix `A` into an upper triangular matrix, while applying the same
    operations to the right-hand side vector `b`. This is the first step in
    solving a linear system of equations of the form ``Ax = b`` using Gaussian
    elimination.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector.
    verbose : bool, optional
        If ``True``, prints detailed information about each elimination step,
        including the row operations performed and the intermediate forms of
        `A` and `b`. Default is ``False``.

    Returns
    -------
    None
        This function modifies `A` and `b` **in place** and does not return
        anything.
    """
    # find shape of system
    n = system_size(A, b)

    # perform forwards elimination
    for i in range(n - 1):
        # eliminate column i
        if verbose:
            print(f"eliminating column {i}")
        for j in range(i + 1, n):
            # row j
            factor = A[j, i] / A[i, i]
            if verbose:
                print(f"  row {j} |-> row {j} - {factor} * row {i}")
            row_add(A, b, j, -factor, i)

        if verbose:
            print()
            print("new system")
            print_array(A)
            print_array(b)
            print()