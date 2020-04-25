import sys
import numpy as np
from scipy.linalg import toeplitz 

def generate(N = 1000, diag = 2.005, sub_diag = -1, T = np.float64):
    elements = np.zeros(N, T)
    elements[0] = diag
    elements[1] = sub_diag
    return toeplitz(elements), np.ones(N, T)

if __name__ == "__main__":
    A = generate()

    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        np.savetxt(filepath, A)