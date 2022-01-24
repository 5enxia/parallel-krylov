import numpy as np
from numpy import float64, dot
from numpy.linalg import norm

from .common import start, finish, init, MultiCpu


def mrr(comm, local_A, b, x=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None) -> tuple:
    # MPI初期化
    rank = comm.Get_rank()
    MultiCpu.joint_mpi(comm)

    # 初期化
    T = float64
    x, maxiter, b_norm, N, residual, num_of_solution_updates = init(
        b, x, maxiter)
    MultiCpu.alloc(local_A, T)
    Ax = np.zeros(N, T)
    Ar = np.zeros(N, T)
    s = np.zeros(N, T)
    rs = np.zeros(1, T)
    ss = np.zeros(1, T)
    nu = np.zeros(1, T)
    mu = np.zeros(1, T)

    # 初期残差
    MultiCpu.dot(local_A, x, out=Ax)
    r = b - Ax
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR + MPI')
    MultiCpu.dot(local_A, r, out=Ar)
    rs = dot(r, Ar)
    ss = dot(Ar, Ar)
    zeta = rs / ss
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z

    i = 1
    num_of_solution_updates[1] = 1

    # 反復計算
    while i < maxiter:
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = residual[i] < tol
        if isConverged:
            break

        # 解の更新
        MultiCpu.dot(local_A, r, out=Ar)
        nu = dot(y, Ar)
        mu = dot(y, y)
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
        return elapsed_time, num_of_solution_updates[:i+1], residual[:i+1]
    else:
        exit(0)
