import sys

import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.process.cpu.common import init, start, end, matvec, vecvec 


def cg(A, b, epsilon, T=np.float64):
    # 初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - matvec(A, x, comm)  # dot
    p = r.copy()

    # 反復計算
    if rank == 0:
        start_time = start(method_name='CG')
    for i in range(max_iter):
        # 収束判定
        residual[i] = norm(r) / b_norm
        isConverged = np.array([residual[i] < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 解の更新
        alpha = vecvec(r, p, comm) / vecvec(p, matvec(A, p, comm), comm)  # dot
        x += alpha * p
        old_r = r.copy()
        r -= alpha * matvec(A, p, comm)  # dot
        beta = vecvec(r, r, comm) / vecvec(old_r, old_r, comm)  # dot
        p = r + beta * p
        num_of_solution_updates[i] = i + 1

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i
    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)


if __name__ == "__main__":
    A, b, epsilon, k, T = getConditionParams('condition.json')
    cg(A, b, epsilon, T)
