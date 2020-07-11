import sys

import cupy as cp
from cupy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.mpi.common import start, end
from krylov.method.mpi.gpu.common import init, matvec, vecvec, vecmat 

def cg(A, b, epsilon, callback = None, T = cp.float64):
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
        
        isConverged = cp.array([residual[i+1] < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
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
    import json
    from krylov.util import loader, toepliz_matrix_generator

    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    with open('condition.json') as f:
        params = json.load(f)
    f.close()

    T = cp.float64
    epsilon = params['epsilon']
    N = params['N'] 
    diag = params['diag']

    A ,b = toepliz_matrix_generator.generate(N=N, diag=diag, T=T)
    A, b = cupy.asarray(A), cupy.asarray(b)
    cg(A, b, epsilon, T)