import numpy as np
from scipy.linalg import toeplitz

T = np.float64
N = 1081
elements = np.zeros(N, T)
elements[0] = 2
elements[1] = 0.01 

from refactor.cpu.cg import cg
A, b = toeplitz(elements), np.ones(N, T)
cg(A, b, 1e-10, T)

# from refactor.cpu.mpi.cg import cg
# A, b = toeplitz(elements), np.ones(N, T)
# cg(A, b, 1e-10, T)