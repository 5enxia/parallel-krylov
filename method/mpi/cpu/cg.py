import sys
import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.mpi.cpu.common import init, start, end, matvec, vecvec, vecmat 

def cg(A, b, epsilon, callback = None, T = np.float64):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    x, y, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)

    if rank == 0:
        start_time = start(method_name = sys._getframe().f_code.co_name)
    
    r = b - matvec(A,x,comm) # dot 
    residual[0] = norm(r) / b_norm
    p = r.copy()


    for i in range(0, max_iter):
        alpha = vecvec(r,p,comm) / vecvec(p,matvec(A,p,comm),comm) # dot
        x += alpha * p
        old_r = r.copy()
        r -= alpha * matvec(A,p,comm) # dot

        residual[i+1] = norm(r) / b_norm
        solution_updates[i] = i + 1
        if residual[i+1] < epsilon:
            isConverged = True
            break

        beta = vecvec(r,r,comm) / vecvec(old_r, old_r,comm) # dot
        p = r + beta * p

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = num_of_iter
    
    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)

    return isConverged


if __name__ == "__main__":
    from krylov.util import loader, toepliz_matrix_generator

    T = np.float64
    epsilon = 1e-8
    N = 1000

    A ,b = toepliz_matrix_generator.generate(N=N, diag=2.005)
    cg(A, b, epsilon, T)
    