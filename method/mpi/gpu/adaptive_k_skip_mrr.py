import sys

import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')

from krylov.method.mpi.common import start, end
from krylov.method.mpi.gpu.common import init, matvec, vecvec, vecmat 

def adaptive_k_skip_mrr(A, b, k, epsilon, callback = None, T = np.float64):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    x, y, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)

    if rank == 0:
        start_time = start(method_name = 'adaptive k-skip MrR', k = k)

    # ================ proto ================ #
    _k_history = list() 
    # ======================================= #

    Ar = np.empty((k+3, N), T)
    Ar[0] = b - matvec(A,x,comm) # dot
    residual[0] = norm(Ar[0]) / b_norm
    pre = residual[0]
    Ay = np.empty((k + 2, N), T)

    # ============== first iter ============= #
    Ar[1] = matvec(A,Ar[0],comm) # dot
    zeta = matvec(Ar[0],Ar[1],comm) / vecvec(Ar[1],Ar[1],comm) # dot
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    # ======================================= #

    alpha = np.empty(2 * k + 3, T)
    beta = np.empty(2 * k + 2, T)
    delta = np.empty(2 * k + 1, T)
    beta[0] = 0

    solution_updates[1] = 1
    dif = 0

    for i in range(1, max_iter):

        rrr = norm(Ar[0]) / b_norm

        if rrr > pre:
            x = pre_x.copy()
            Ar[0] = b - matvec(A,x,comm) # dot
            Ar[1] = matvec(A,Ar[0],comm) # dot
            zeta = vecvec(Ar[0],Ar[1],comm) / vecvec(Ar[1],Ar[1],comm) # dot
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            if k > 1:
                dif += 1
                k -= 1

        else:
            pre = rrr
            residual[i - dif] = rrr
            pre_x = x.copy()
            
        # ================ proto ================ #
        _k_history.append(k) 
        # ======================================= #

        isConverged = np.array([rrr < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        for j in range(1, k + 2):
            Ar[j] = matvec(A,Ar[j-1],comm) # dot

        for j in range(1, k + 1):
            Ay[j] = matvec(A,Ay[j-1],comm) # dot

        for j in range(2 * k + 3):
            jj = j // 2
            alpha[j] = vecvec(Ar[jj],Ar[jj+j%2],comm) # dot

        for j in range(1, 2 * k + 2):
            jj = j // 2
            beta[j] = vecvec(Ay[jj],Ar[jj+j%2],comm) # dot

        for j in range(0, 2 * k + 1):
            jj = j // 2
            delta[j] = vecvec(Ay[jj],Ay[jj+j%2],comm) # dot

        sigma = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / sigma
        eta = -alpha[1] * beta[1] / sigma

        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = matvec(A,Ar[0],comm) # dot
        x -= z

        for j in range(0, k):
            delta[0] = zeta ** 2 * alpha[2] + eta * zeta * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = eta ** 2 * delta[1] + 2 * eta * zeta * beta[2] + zeta ** 2 * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]

            for l in range(2, 2 * (k - j) + 1):
                delta[l] = eta ** 2 * delta[l] + 2 * eta * zeta * beta[l + 1] + zeta ** 2 *alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]

            sigma = alpha[2] * delta[0] - beta[1 ]** 2
            zeta = alpha[1] * delta[0] / sigma
            eta = -alpha[1] * beta[1] / sigma

            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = matvec(A,Ar[0],comm) # dot? 
            x -= z

        solution_updates[i + 1 - dif] = solution_updates[i - dif] + k + 1

    else:
        isConverged = False
        
    num_of_iter = i + 1
    residual_index = i - dif

    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = k)

    return isConverged



if __name__ == "__main__":
    from krylov.util import loader, toepliz_matrix_generator
    import json
    
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    with open('condition.json') as f:
        params = json.load(f)
    f.close()

    T = np.float64
    epsilon = params['epsilon']
    N = params['N'] 
    diag = params['diag']
    k = params['k']

    A, b = toepliz_matrix_generator.generate(N=N,diag=diag,T=T)
    adaptive_k_skip_mrr(A,b,k,epsilon,T)
