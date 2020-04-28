import sys
import numpy as np
from mpi4py import MPI

sys.path.append('../../../')
from krylov.util.toepliz_matrix_generator import generate4mpi
from krylov.method.mpi.common import start, end, dot,init

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_of_process = comm.Get_size()

    N = 10
    A, b = generate4mpi(rank, N = N)
    result = dot(A,b,np.empty(N),comm, N, num_of_process)
    
    if rank == 0:
        print(f'{rank}:{result}')