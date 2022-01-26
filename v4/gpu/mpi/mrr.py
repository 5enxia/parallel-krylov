from numpy import float64
import cupy as cp
from cupy import dot
from cupy.linalg import norm
from mpi4py import MPI

from .common import start, finish, init, MultiGpu


def mrr(comm, local_A, b, x=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None) -> tuple:
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
    Ar = cp.zeros(N, T)

    # 初期残差
    MultiGpu.dot(local_A, x, out=Ax)
    r = b - Ax
    residual[0] = norm(r) / b_norm

    # 初期反復
    i = 0
    if rank == 0:
        start_time = start(method_name='MrR + GPU + MPI')
    MultiGpu.dot(local_A, r, out=Ar)
    zeta = dot(r, Ar) / dot(Ar, Ar)
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1
    i += 1

    # 反復計算
    while i < maxiter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < tol:
            isConverged = True
            break

        # 解の更新
        MultiGpu.dot(local_A, r, out=Ar)
        mu = dot(y, y)
        nu = dot(y, Ar)
        gamma = nu / mu
        s = Ar - gamma * y
        rs = dot(r, s)
        ss = dot(s, s)
        zeta = rs / ss
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
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
