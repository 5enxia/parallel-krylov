import numpy as np
from numpy import dot
from numpy.linalg import norm

from ..common import start, end
from .common import init


def adaptive_k_skip_mrr(A: np.ndarray, b: np.ndarray, epsilon: float, k: int, T=np.float64):
    """[summary]

    Args:
        A (np.ndarray): 係数行列A
        b (np.ndarray): bベクトル
        epsilon (float): 収束判定子
        k (int): k
        T ([type], optional): 浮動小数精度 Defaults to np.float64.

    Returns:
        float: 経過時間
        np.ndarray: 残差更新履歴
        np.ndarray: 残差履歴
    """
    # 初期化
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)
    Ar = np.empty((k+3, N), T)
    Ay = np.empty((k + 2, N), T)
    alpha = np.empty(2 * k + 3, T)
    beta = np.empty(2 * k + 2, T)
    delta = np.empty(2 * k + 1, T)
    beta[0] = 0
    dif = 0
    k_history = np.zeros(N+1, np.int)
    k_history[0] = k

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
    num_of_solution_updates[1] = 1
    k_history[1] = k
    i = 1
    index = 1

    # 反復計算
    while i < max_iter:
        cur_residual = norm(Ar[0]) / b_norm

        # 残差減少判定
        if cur_residual > pre_residual:
            # 残差と解をk+1反復前の状態に戻す
            x = pre_x.copy()
            Ar[0] = b - dot(A, x)
            Ar[1] = dot(A, Ar[0])
            zeta = dot(Ar[0], Ar[1]) / dot(Ar[1], Ar[1])
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z
            # kを1下げる
            if k > 1:
                k -= 1
                # dif += 1
        else:
            pre_residual = cur_residual
            residual[index] = cur_residual
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
        for j in range(k):
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

        i += (k + 1)
        index += 1
        num_of_solution_updates[index] = i
        k_history[index] = k
    else:
        isConverged = False

    elapsed_time = end(start_time, isConverged, i, residual[index], k)
    return elapsed_time, num_of_solution_updates[:index+1], residual[:index+1], k_history[:index+1]
