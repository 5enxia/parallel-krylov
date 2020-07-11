import sys
import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')
    
from krylov.method.mpi.cpu.common import init, start, end, matvec, vecvec, vecmat

def mrr(A, b, epsilon, callback = None, T = np.float64):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    x, y, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)

    if rank == 0:
        start_time = start(method_name = 'MrR')
    
    r = np.zeros(max_iter, T)
    r = b - matvec(A,x,comm) # dot
    residual[0] = norm(r) / b_norm
    z = np.zeros(N, T)

    # ======= first iter ====== #
    Ar = matvec(A,r,comm) # dot
    zeta = vecvec(r,Ar,comm) / vecvec(Ar,Ar,comm) # dot
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    solution_updates[1] = 1
    # ========================= #

    for i in range(1, max_iter):
        
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        Ar = matvec(A,r,comm) # dot
        nu = vecvec(y,Ar,comm) # dot
        gamma = nu / vecvec(y,y,comm) # dot
        s = Ar - gamma * y
        zeta = vecvec(r,s,comm) / vecvec(s,s,comm) # dot
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        
        solution_updates[i] = i + 1
    
    else:
        isConverged = False
    
    num_of_iter = i + 1
    residual_index = i 

    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)
    
    return isConverged

    
if __name__ == "__main__":
    from krylov.util import loader, toepliz_matrix_generator
    T = np.float64
    epsilon = 1e-8
    N = 1000

    A ,b = toepliz_matrix_generator.generate(N=N, diag=2.005)
    mrr(A, b, epsilon, T)