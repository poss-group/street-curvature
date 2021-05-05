import numpy as np
import matplotlib.pyplot as plt

def calc_row_idx(k, n):
    return int(np.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    """
    Get the square matrix indices from a condensed distance matrix index.

    Parameters
    ----------
    k : int
        Index of condensed matrix
    n : int
        size of square matrix

    Returns
    -------
    i, j : ints
        Indices in square matrix
    """
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j
