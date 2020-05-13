import sys
import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.mpi.cpu.common import init, start, end, matvec, vecvec, vecmat 


if __name__ == "__main__":
    # condition
    N = 10
    T = np.float64
    
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    NUM_OF_PROCESS = comm.Get_size()

    if rank == 0:
        A, b = generate(N=N,T=T)
    else:
        A = None
        b = np.zeros(N, dtype=T)

    y = np.zeros(N, dtype=T)

    # transfer data
    num_of_local_row = N // NUM_OF_PROCESS
    local_A = np.zeros((num_of_local_row, N), dtype=T)
    comm.Bcast(b, root=0)
    comm.Scatter(A, local_A, root=0)

    # check transfered data 
    if rank == 0:
        print(A)

    print(f'{rank}:{local_A}')

    # calc
    local_y = np.dot(local_A,b)

    # transfer calclated data
    comm.Gather(local_y, y, root=0)

    # check transfered data 
    if rank == 0:
        print(y)