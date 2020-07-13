import sys

import numpy as np
from numpy import dot
from numpy.linalg import norm
from mpi4py import MPI

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.process.cpu.common import init, start, end, matvec, vecvec 


def k_skip_cg(A, b, epsilon, k, T=np.float64):
    # 初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_of_process = comm.Get_size()
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)
    # root 
    Ax = np.empty(N, T)  # matvec演算結果格納用
    Ar = np.zeros((k + 2, N), T)
    Ap = np.zeros((k + 3, N), T)
    a = np.zeros(2 * k + 2, T)
    f = np.zeros(2 * k + 4, T)
    c = np.zeros(2 * k + 2, T)
    # local
    local_N = N // num_of_process
    local_A = np.empty((local_N, N), T)
    local_Ax = np.empty(local_N, T)  # matvec演算結果格納用
    local_Ar = np.zeros((k + 2, local_N), T)
    local_Ap = np.zeros((k + 3, local_N), T)
    local_a = np.zeros(2 * k + 2, T)
    local_f = np.zeros(2 * k + 4, T)
    local_c = np.zeros(2 * k + 2, T)

    # 初期残差
    Ar[0] = b - matvec(A, local_A, x, Ax, local_Ax, comm)
    Ap[0] = Ar[0]

    # 反復計算
    if rank == 0:
        start_time = start(method_name='k-skip CG', k=k)
    for i in range(max_iter):
        # 収束判定
        residual[i] = norm(Ar[0]) / b_norm
        isConverged = np.array([residual[i] < epsilon], dtype=bool)
        comm.Bcast(isConverged, root=0)
        if isConverged:
            break

        # 事前計算
        comm.Scatter(A, local_A, root=0)
        for j in range(1, k + 1):
            comm.Bcast(Ar[j-1], root=0)
            local_Ar = dot(local_A, Ar[j-1])
            comm.Gather(local_Ar, Ar[j], root=0)
        for j in range(1, k + 2):
            comm.Bcast(Ap[j-1], root=0)
            local_Ap = dot(local_A, Ap[j-1])
            comm.Gather(local_Ap, Ap[j], root=0)
        comm.Bcast(Ar)
        comm.Bcast(Ap)
        for j in range(2 * k + 1):
            jj = j // 2
            local_a[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ar[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_a, a, root=0)
        for j in range(2 * k + 4):
            jj = j // 2
            local_f[j] = dot(
                Ap[jj][rank * local_N: (rank+1) * local_N],
                Ap[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_f, f, root=0)
        for j in range(2 * k + 2):
            jj = j // 2
            local_c[j] = dot(
                Ar[jj][rank * local_N: (rank+1) * local_N],
                Ap[jj + j % 2][rank * local_N: (rank+1) * local_N]
            )
        comm.Reduce(local_c, c, root=0)

        # CGでの1反復
        alpha = a[0] / f[1]
        beta = alpha ** 2 * f[2] / a[0] - 1
        x += alpha * Ap[0]
        Ar[0] -= alpha * Ap[1]
        Ap[0] = Ar[0] + beta * Ap[0]
        Ap[1] = matvec(A, local_A, Ap[0], Ax, local_Ax, comm)

        # CGでのk反復
        for j in range(k):
            for l in range(0, 2*(k-j)+1):
                a[l] += alpha*(alpha*f[l+2] - 2*c[l+1])
                d = c[l] - alpha*f[l+1]
                c[l] = a[l] + d*beta
                f[l] = c[l] + beta*(d + beta*f[l])

            # 解の更新
            alpha = a[0] / f[1]
            beta = alpha ** 2 * f[2] / a[0] - 1
            x += alpha * Ap[0]
            Ar[0] -= alpha * Ap[1]
            Ap[0] = Ar[0] + beta * Ap[0]
            Ap[1] = matvec(A, local_A, Ap[0], Ax, local_Ax, comm)

        num_of_solution_updates[i+1] = num_of_solution_updates[i] + k + 1

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i
    if rank == 0:
        end(start_time, isConverged, num_of_iter, residual, residual_index)


if __name__ == "__main__":
    A, b, epsilon, k, T = getConditionParams('condition.json')
    k_skip_cg(A, b, epsilon, k, T)
