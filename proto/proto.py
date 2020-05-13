import sys
import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.mpi.cpu.common import init, start, end, matvec, vecvec, vecmat
from krylov.util.toepliz_matrix_generator import generate

if __name__ == "__main__":
    # condition
    T = np.float64
    N = 10
    
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # matvec 
    if rank == 0:
        A, b = generate(N=N,T=T)
    else:
        A = np.zeros((N,N), dtype=T) 
        b = np.zeros(N, dtype=T)
    y = matvec(A,b,comm,T)

    if rank == 0:
        print(y)

    # main 2 
    y = vecvec(b,b,comm,T)

    if rank == 0:
        print(y)