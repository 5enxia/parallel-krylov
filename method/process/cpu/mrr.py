import sys

import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.process.cpu.common import init, start, end, matvec, vecvec 


def mrr(A, b, epsilon, T=np.float64):
    # 初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - matvec(A, x, comm)  # dot
    residual[0] = norm(r) / b_norm

    # 初期反復
    if rank == 0:
        start_time = start(method_name='MrR')
    Ar = matvec(A, r, comm)  # dot
    zeta = vecvec(r, Ar, comm) / vecvec(Ar, Ar, comm)  # dot
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1

    # 反復計算
    for i in range(1, max_iter):
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = np.array([residual[i] < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 解の更新
        Ar = matvec(A, r, comm)  # dot
        nu = vecvec(y, Ar, comm)  # dot
        gamma = nu / vecvec(y, y, comm)  # dot
        s = Ar - gamma * y
        zeta = vecvec(r, s, comm) / vecvec(s, s, comm)  # dot
        eta = -zeta * gamma
        y = eta * y + zeta * Ar
        z = eta * z - zeta * r
        r -= y
        x -= z
        num_of_solution_updates[i] = i + 1
    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i
    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)

    
if __name__ == "__main__":
    A, b, epsilon, k, T = getConditionParams('condition.json')
    mrr(A, b, epsilon, T)
