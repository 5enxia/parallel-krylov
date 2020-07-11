import sys

import numpy as np
from numpy import dot
from numpy.linalg import norm

if __name__ == "__main__":
    sys.path.append('../../../../')
    from krylov.method.common import getConditionParams
    from krylov.method.threads.common import start, end
    from krylov.method.threads.cpu.common import init


def mrr(A, b, epsilon, T=np.float64):
    # 初期化
    x, b_norm, N, max_iter, residual, num_of_solution_updates = init(A, b, T)

    # 初期残差
    r = b - dot(A, x)
    residual[0] = norm(r) / b_norm

    # 初期反復
    start_time = start(method_name='MrR')
    Ar = dot(A, r)
    zeta = dot(r, Ar) / dot(Ar, Ar)
    y = zeta * Ar
    z = -zeta * r
    r -= y
    x -= z
    num_of_solution_updates[1] = 1

    # 反復計算
    for i in range(1, max_iter):
        # 収束判定
        residual[i] = norm(r) / b_norm
        if residual[i] < epsilon:
            isConverged = True
            break

        # 解の更新
        Ar = dot(A, r)
        gamma = dot(y, Ar) / dot(y, y)
        s = Ar - gamma * y
        zeta = dot(r, s) / dot(s, s)
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
    end(start_time, isConverged, num_of_iter, residual, residual_index)


if __name__ == "__main__":
    A, b, epsilon, k, T = getConditionParams('condition.json')
    mrr(A, b, epsilon, T)
