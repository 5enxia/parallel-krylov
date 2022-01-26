import numpy as np
from numpy import float64, dot
from numpy.linalg import norm

from .common import start, finish, init, MultiCpu

def cg(comm, local_A, b, x=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None) -> tuple:
    # MPI初期化
    rank = comm.Get_rank()
    MultiCpu.joint_mpi(comm)

    # 初期化
    T = float64
    x, maxiter, b_norm, N, residual, num_of_solution_updates = init(b, x, maxiter)
    MultiCpu.alloc(local_A, T)
    Ax = np.zeros(N, T)
    v = np.zeros(N, T)

    # 初期残差
    MultiCpu.dot(local_A, x, out=Ax)
    r = b - Ax
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0

    if rank == 0:
        start_time = start(method_name='CG + MPI')
    while i < maxiter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < tol:
            isConverged = True 
            break

        # 解の更新
        MultiCpu.dot(local_A, p, out=v)
        sigma = dot(p, v)
        alpha = gamma / sigma
        x += alpha * p
        r -= alpha * v
        old_gamma = gamma.copy()
        gamma = dot(r, r)
        beta = gamma / old_gamma
        p = r + beta * p
        i += 1
        num_of_solution_updates[i] = i
    else:
        isConverged = False
        residual[i] = norm(r) / b_norm

    if rank == 0:
        elapsed_time = finish(start_time, isConverged, i, residual[i])
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
