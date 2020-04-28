import sys

import numpy as np 

sys.path.append('../../../')
from krylov.util import loader, toepliz_matrix_generator
from krylov.method.mpi.cg import cg

if __name__ == "__main__":
    T = np.float64
    epsilon = 1e-8

    A ,b = toepliz_matrix_generator.generate(N=1000, diag=2.5)
    cg(A, b, epsilon, T)