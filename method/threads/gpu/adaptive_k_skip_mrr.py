import sys

import cupy as cp
from cupy import dot
from cupy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.threads.common import start, end
    from krylov.method.threads.gpu.common import init


def adaptive_k_skip_mrr(A, b, epsilon, k, T=cp.float64):
    # 初期化
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)
    Ar = cp.empty((k+3, N), T)
    Ay = cp.empty((k + 2, N), T)
    alpha = cp.empty(2 * k + 3, T)
    beta = cp.empty(2 * k + 2, T)
    delta = cp.empty(2 * k + 1, T)
    beta[0] = 0
    dif = 0

    # 初期残差
    Ar[0] = b - dot(A, x)
    residual[0] = norm(Ar[0]) / b_norm
    pre_residual = residual[0]

    # 初期反復
    start_time = start(method_name='Adaptive k-skip MrR', k=k)
    Ar[1] = dot(A, Ar[0])
    zeta = dot(Ar[0], Ar[1]) / dot(Ar[1], Ar[1])
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    solution_updates[1] = 1

    # 反復計算
    for i in range(1, max_iter):
        cur_residual = norm(Ar[0]) / b_norm
        # 残差減少判定
        if cur_residual > pre_residual:
            # 残差と解を直前の状態に戻す
            x = pre_x.copy()
            Ar[0] = b - dot(A, x)
            Ar[1] = dot(A, Ar[0])
            zeta = dot(Ar[0], Ar[1]) / dot(Ar[1], Ar[1])
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            # kを下げて収束を安定化させる
            if k > 1:
                k -= 1
                dif += 1
        else:
            pre_residual = cur_residual
            residual[i - dif] = cur_residual
            pre_x = x.copy()

        # 収束判定
        if cur_residual < epsilon:
            isConverged = True
            break

        # 事前計算
        for j in range(1, k + 2):
            Ar[j] = dot(A, Ar[j-1])
        for j in range(1, k + 1):
            Ay[j] = dot(A, Ay[j-1])
        for j in range(2 * k + 3):
            jj = j // 2
            alpha[j] = dot(Ar[jj], Ar[jj + j % 2])
        for j in range(1, 2 * k + 2):
            jj = j // 2
            beta[j] = dot(Ay[jj], Ar[jj + j % 2])
        for j in range(0, 2 * k + 1):
            jj = j // 2
            delta[j] = dot(Ay[jj], Ay[jj + j % 2])

        # MrRでの1反復(解の更新)
        sigma = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / sigma
        eta = -alpha[1] * beta[1] / sigma
        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = dot(A, Ar[0])
        x -= z

        # MrRでのk反復
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

            # 解の更新
            sigma = alpha[2] * delta[0] - beta[1] ** 2
            zeta = alpha[1] * delta[0] / sigma
            eta = -alpha[1] * beta[1] / sigma
            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = dot(A, Ar[0])
            x -= z

        solution_updates[i + 1 - dif] = solution_updates[i - dif] + k + 1

    else:
        isConverged = False

    num_of_iter = i + 1
    residual_index = i - dif
    end(start_time, isConverged, num_of_iter, residual, residual_index, final_k=k)


if __name__ == "__main__":
    A, b, epsilon, k, T = getConditionParams('condition.json')
    adaptive_k_skip_mrr(A, b, epsilon, k, T)
