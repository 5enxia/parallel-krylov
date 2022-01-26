from numpy import float64
import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

from .common import start, finish, init, MultiGpu


def cg(comm, local_A, b, x=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None) -> tuple:
    # MPI初期化
    rank = comm.Get_rank()
    MultiGpu.joint_mpi(comm)

    # 初期化
    T = float64
    ## GPU初期化
    MultiGpu.init()
    b, x, maxiter, b_norm, N, residual, num_of_solution_updates = init(b, x, maxiter)
    MultiGpu.alloc(local_A, b, T)
    Ax = cp.zeros(N, T)
    v = cp.zeros(N, T)

    # 初期残差
    MultiGpu.dot(local_A, x, out=Ax)
    r = b - Ax
    p = r.copy()
    gamma = dot(r, r)

    # 反復計算
    i = 0

    if rank == 0:
        start_time = start(method_name='CG + GPU + MPI')

    while i < maxiter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < tol:
            isConverged = True
            break

        # 解の更新
        MultiGpu.dot(local_A, p, out=v)
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
        info = {
            'time': elapsed_time,
            'nosl': num_of_solution_updates[:i+1],
            'residual': residual[:i+1],
        }
        return x, info
    else:
        exit(0)
