import numpy as np
from scipy.linalg import toeplitz


def generate(N, diag, sub_diag, T):
    elements = np.zeros(N, T)
    elements[0] = diag
    elements[1] = sub_diag
    return toeplitz(elements), np.ones(N, T)
