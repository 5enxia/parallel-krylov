import numpy as np

from .common import start, end as finish, init, init_mpi, krylov_base_start, krylov_base_finish


def _adaptivekskipmrr_cpu(A, b, epsilon, k, T, pu):
    from numpy.linalg import norm

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = np.empty(N, T)
    Ar = np.empty((k + 2, N), T)
    Ay = np.empty((k + 1, N), T)
    rAr = np.zeros(1, T)
    ArAr = np.zeros(1, T)
    alpha = np.zeros(2*k + 3, T)
    beta = np.zeros(2*k + 2, T)
    delta = np.zeros(2*k + 1, T)
    # local
    local_alpha = np.zeros(2*k + 3, T)
    local_beta = np.zeros(2*k + 2, T)
    local_delta = np.zeros(2*k + 1, T)
    # kの履歴
    k_history = np.zeros(max_iter+1, np.int)
    k_history[0] = k

    # 初期残差
    comm.Allgather(A[begin:end].dot(x), Ax)
    Ar[0] = b - Ax
    residual[0] = norm(Ar[0]) / b_norm

    # 残差減少判定変数
    cur_residual = residual[0].copy()
    pre_residual = residual[0].copy()

    # 初期反復
    if rank == 0:
        start_time = start(method_name='adaptive k-skip MrR', k=k)
    comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
    comm.Allreduce(Ar[0][begin:end].dot(Ar[1][begin:end]), rAr)
    comm.Allreduce(Ar[1][begin:end].dot(Ar[1][begin:end]), ArAr)
    zeta = rAr / ArAr
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z

    i = 1
    index = 1
    num_of_solution_updates[1] = 1
    k_history[1] = k

    # 反復計算
    while i < max_iter:
        pre_residual = cur_residual
        cur_residual = norm(Ar[0]) / b_norm
        residual[index] = cur_residual

        # 残差減少判定
        if cur_residual > pre_residual:
            # 解と残差を再計算
            x = pre_x.copy()

            comm.Allgather(A[begin:end].dot(x), Ax)
            comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
            comm.Allreduce(Ar[0][begin:end].dot(Ar[1][begin:end]), rAr)
            comm.Allreduce(Ar[1][begin:end].dot(Ar[1][begin:end]), ArAr)
            zeta = rAr / ArAr
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            i += 1
            index += 1
            num_of_solution_updates[index] = i
            residual[index] = norm(Ar[0]) / b_norm

            # kを下げて収束を安定化させる
            if k > 1:
                k -= 1
            k_history[index] = k
        else:
            pre_x = x.copy()
            
        # 収束判定
        isConverged = cur_residual < epsilon
        if isConverged:
            break

        # 基底計算
        for j in range(1, k + 1):
            comm.Allgather(A[begin:end].dot(Ar[j-1]), Ar[j])
            comm.Allgather(A[begin:end].dot(Ay[j-1]), Ay[j])
        comm.Allgather(A[begin:end].dot(Ar[k]), Ar[k+1])

        # 係数計算
        local_alpha[0] = Ar[0][begin:end].dot(Ar[0][begin:end])
        local_delta[0] = Ay[0][begin:end].dot(Ay[0][begin:end])
        for j in range(1, 2*k+1):
            jj = j//2
            local_alpha[j] = Ar[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
            local_beta[j] = Ay[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
            local_delta[j] = Ay[jj][begin:end].dot(Ay[jj + j % 2][begin:end])
        local_alpha[2*k+1] = Ar[k][begin:end].dot(Ar[k+1][begin:end])
        local_beta[2*k+1] = Ay[k][begin:end].dot(Ar[k+1][begin:end])
        local_alpha[2*k+2] = Ar[k+1][begin:end].dot(Ar[k+1][begin:end])
        comm.Allreduce(local_alpha, alpha)
        comm.Allreduce(local_beta, beta)
        comm.Allreduce(local_delta, delta)

        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        x -= z

        # MrRでのk反復
        for j in range(k):
            zz = zeta ** 2
            ee = eta ** 2
            ez = eta * zeta
            delta[0] = zz * alpha[2] + ez * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = ee * delta[1] + 2 * eta * zeta * beta[2] + zz * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = ee * delta[l] + 2 * ez * beta[l+1] + zz * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解と残差の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
        k_history[index] = k
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    num_of_iter = i
    if rank == 0:
        elapsed_time = finish(start_time, isConverged, num_of_iter, residual[index], k)
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1]
    else:
        exit(0)


def _adaptivekskipmrr_gpu(A, b, epsilon, k, T, pu):
    import cupy as cp
    from cupy.linalg import norm

    from .common import init_gpu

    # 共通初期化
    comm, rank, num_of_process = init_mpi()
    init_gpu(rank)
    A, b, x, b_norm, N, local_N, max_iter, residual, num_of_solution_updates = init(A, b, num_of_process, T, pu)
    begin, end = rank * local_N, (rank+1) * local_N

    # 初期化
    Ax = np.empty(N, T)
    Ar = cp.zeros((k + 2, N), T)
    Ay = cp.zeros((k + 1, N), T)
    rAr = cp.empty(1, T)
    ArAr = cp.empty(1, T)
    alpha = cp.zeros(2*k + 3, T)
    beta = cp.zeros(2*k + 2, T)
    delta = cp.zeros(2*k + 1, T)
    # cpu
    Ar_cpu = np.zeros((k + 2, N), T)
    Ay_cpu = np.zeros((k + 1, N), T)
    rAr_cpu = np.empty(1, T)
    ArAr_cpu = np.empty(1, T)
    alpha_cpu = np.zeros(2*k + 3, T)
    beta_cpu = np.zeros(2*k + 2, T)
    delta_cpu = np.zeros(2*k + 1, T)
    # kの履歴
    k_history = np.zeros(max_iter+1, np.int)
    k_history[0] = k

    krylov_base_times = np.zeros(max_iter, T)  # time

    # 初期残差
    comm.Allgather(A[begin:end].dot(x).get(), Ax)
    Ar[0] = b - cp.asarray(Ax)
    residual[0] = norm(Ar[0]) / b_norm

    # 残差現象判定変数
    cur_residual = residual[0].copy()
    pre_residual = residual[0].copy()

    # 初期反復
    if rank == 0:
        start_time = start(method_name='adaptive k-skip MrR', k=k)

    krylov_base_times[0] = krylov_base_start()  # time

    Ar[1][begin:end] = A[begin:end].dot(Ar[0])
    comm.Allgather(Ar[1][begin:end].get(), Ar_cpu[1])
    Ar[1] = cp.asarray(Ar_cpu[1])
    comm.Allreduce(Ar[0][begin:end].dot(Ar[1][begin:end]).get(), rAr_cpu)
    comm.Allreduce(Ar[1][begin:end].dot(Ar[1][begin:end]).get(), ArAr_cpu)
    rAr = cp.asarray(rAr_cpu)
    ArAr = cp.asarray(ArAr_cpu)
    zeta = rAr / ArAr
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z

    krylov_base_times[0] = krylov_base_finish(krylov_base_times[0])  # time

    i = 1
    index = 1
    num_of_solution_updates[1] = 1
    k_history[1] = k

    # 反復計算
    while i < max_iter:
        pre_residual = cur_residual
        cur_residual = norm(Ar[0]) / b_norm
        residual[index] = cur_residual

        # 残差減少判定
        if cur_residual > pre_residual:
            # 解と残差を再計算
            x = pre_x.copy()

            comm.Allgather(A[begin:end].dot(x).get(), Ax)
            Ar[0] = b - cp.asarray(Ax)
            Ar[1][begin:end] = A[begin:end].dot(Ar[0])
            comm.Allgather(Ar[1][begin:end].get(), Ar_cpu[1])
            Ar[1] = cp.asarray(Ar_cpu[1])

            krylov_base_times[index] = krylov_base_start()  # time

            comm.Allreduce(Ar[0][begin:end].dot(Ar[1][begin:end]).get(), rAr_cpu)
            comm.Allreduce(Ar[1][begin:end].dot(Ar[1][begin:end]).get(), ArAr_cpu)

            krylov_base_times[index] = krylov_base_finish(krylov_base_times[index])  # time

            rAr = cp.asarray(rAr_cpu)
            ArAr = cp.asarray(ArAr_cpu)
            zeta = rAr / ArAr
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            i += 1
            index += 1
            num_of_solution_updates[index] = i
            residual[index] = norm(Ar[0]) / b_norm
            
            # kを下げて収束を安定化させる
            if k > 1:
                k -= 1
            k_history[index] = k
        else:
            pre_x = x.copy()
            
        # 収束判定
        isConverged = cur_residual < epsilon
        if isConverged:
            break

        # 基底計算
        for j in range(1, k + 1):
            comm.Allgather(A[begin:end].dot(Ar[j-1]).get(), Ar_cpu[j])
            comm.Allgather(A[begin:end].dot(Ay[j-1]).get(), Ay_cpu[j])
            Ar[j] = cp.asarray(Ar_cpu[j])
            Ay[j] = cp.asarray(Ay_cpu[j])
        comm.Allgather(A[begin:end].dot(Ar[k]).get(), Ar_cpu[k+1])
        Ar[k+1] = cp.asarray(Ar_cpu[k+1])

        # 係数計算
        alpha[0] = Ar[0][begin:end].dot(Ar[0][begin:end])
        delta[0] = Ay[0][begin:end].dot(Ay[0][begin:end])
        for j in range(1, 2*k+1):
            jj = j//2
            alpha[j] = Ar[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
            beta[j] = Ay[jj][begin:end].dot(Ar[jj + j % 2][begin:end])
            delta[j] = Ay[jj][begin:end].dot(Ay[jj + j % 2][begin:end])
        alpha[2*k+1] = Ar[k][begin:end].dot(Ar[k+1][begin:end])
        beta[2*k+1] = Ay[k][begin:end].dot(Ar[k+1][begin:end])
        alpha[2*k+2] = Ar[k+1][begin:end].dot(Ar[k+1][begin:end])
        comm.Allreduce(alpha.get(), alpha_cpu)
        comm.Allreduce(beta.get(), beta_cpu)
        comm.Allreduce(delta.get(), delta_cpu)
        alpha = cp.asarray(alpha_cpu)
        beta = cp.asarray(beta_cpu)
        delta = cp.asarray(delta_cpu)


        # MrRでの1反復(解と残差の更新)
        d = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / d
        eta = -alpha[1] * beta[1] / d
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        x -= z

        krylov_base_times[index] = krylov_base_start()  # time

        # MrRでのk反復
        for j in range(k):
            zz = zeta ** 2
            ee = eta ** 2
            ez = eta * zeta
            delta[0] = zz * alpha[2] + ez * beta[1]
            alpha[0] -= zeta * alpha[1]
            delta[1] = ee * delta[1] + 2 * eta * zeta * beta[2] + zz * alpha[3]
            beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
            alpha[1] = -beta[1]
            for l in range(2, 2 * (k - j) + 1):
                delta[l] = ee * delta[l] + 2 * ez * beta[l+1] + zz * alpha[l + 2]
                tau = eta * beta[l] + zeta * alpha[l + 1]
                beta[l] = tau - delta[l]
                alpha[l] -= tau + beta[l]
            # 解と残差の更新
            d = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / d
            eta = -alpha[1] * beta[1] / d
            comm.Allgather(A[begin:end].dot(Ar[0]).get(), Ar_cpu[1])
            Ar[1] = cp.asarray(Ar_cpu[1])
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

        krylov_base_times[index] = krylov_base_finish(krylov_base_times[index])  # time

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
        k_history[index] = k
    else:
        isConverged = False
        residual[index] = norm(Ar[0]) / b_norm

    num_of_iter = i
    if rank == 0:
        elapsed_time = finish(start_time, isConverged, num_of_iter, residual[index], k)
        return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1], krylov_base_times[:index+1]
    else:
        exit(0)


def adaptivekskipmrr(A, b, epsilon, k, T, pu):
    comm, rank, num_of_process = init_mpi()
    if pu == 'cpu':
        if rank == 0:
            return _adaptivekskipmrr_cpu(A, b, epsilon, k, T, pu)
        else:
            _adaptivekskipmrr_cpu(A, b, epsilon, k, T, pu)
    else:
        if rank == 0:
            return _adaptivekskipmrr_gpu(A, b, epsilon, k, T, pu)
        else:
            _adaptivekskipmrr_gpu(A, b, epsilon, k, T, pu)
