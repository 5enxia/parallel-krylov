import sys
from ..common import init, start, end 
import numpy as np
from numpy import dot
from numpy.linalg import norm, multi_dot


def variable_k_skip_mrr(A, b, k, epsilon, callback = None, T = np.float64):
    x, b_norm, N, max_iter, residual, solution_updates = init(A, b, T)
    start_time = start(method_name = 'variable k-skip MrR', k = k)
    

    # ================ proto ================ #
    _k_history = list() 
    tmp = k * (max_iter)
    old_div_r = 0
    mid_div_r = 0
    new_div_r = 0
    
    Ar = np.empty(((k + 3) * tmp, N), T)
    Ar[0] = b - dot(A,x)
    residual[0] = norm(Ar[0]) / b_norm
    pre = residual[0]
    Ay = np.empty(((k + 2) * tmp, N), T)
    # ======================================= #

    # ============== first iter ============= #
    Ar[1] = dot(A,Ar[0])
    zeta = dot(Ar[0],Ar[1]) / dot(Ar[0],Ar[1])
    Ay[0] = zeta * Ar[1]
    z = -zeta * Ar[0]
    Ar[0] -= Ay[0]
    x -= z
    # ======================================= #

    # ================ proto ================ #
    alpha = np.empty((2 * k + 3) * tmp, T) 
    beta = np.empty((2 * k + 2) * tmp, T) 
    delta = np.empty((2 * k + 1) * tmp, T) 
    # ======================================= #
    beta[0] = 0

    solution_updates[1] = 1
    dif = 0
    
    # ================ proto ================ #
    count = 0
    # ======================================= #

    for i in range(1, max_iter):
        
        rrr = norm(Ar[0]) / b_norm
        
        # ================ proto ================ #
        new_div_log_r = np.log10(rrr) - np.log10(pre)
        # ======================================= #

        if rrr > pre:

            x = pre_x.copy()
            Ar[0] = b - dot(A,x)
            Ar[1] = dot(A,Ar[0])
            zeta = dot(Ar[0],Ar[1]) / dot(Ar[1],Ar[1])
            Ay[0] = zeta * Ar[1]
            z = -zeta * Ar[0]
            Ar[0] -= Ay[0]
            x -= z

            if k > 1:
                dif += 1
                k -= 1      

        else:
            pre = rrr
            residual[i - dif] = rrr
            pre_x = x.copy()
            
            # ================ proto ================ #
            if k < 8:
                if new_div_log_r < mid_div_r < old_div_r: 
                    if dif > 0:
                        dif -= 1
                    k += 1
            # ======================================= #
        
        # ================ proto ================ #
        old_div_r = mid_div_r
        mid_div_r = new_div_log_r
        _k_history.append(k)
        # ======================================= #


        if rrr < epsilon:
            self._converged(i,i-dif,k)
            isConverged = True
            break

        for j in range(1, k + 2):
            Ar[j] = dot(A,Ar[j-1])

        for j in range(1, k + 1):
            Ay[j] = dot(A,Ay[j-1])

        for j in range(0, 2 * k + 3):
            jj = j // 2
            alpha[j] = dot(Ar[jj],Ar[jj+j%2])

        for j in range(1, 2 * k + 2):
            jj = j // 2
            beta[j] = dot(Ay[jj],Ar[jj+j%2])

        for j in range(0, 2 * k + 1):
            jj = j // 2
            delta[j] = dot(Ay[jj],Ay[jj+j%2])

        sigma = alpha[2] * delta[0] - beta[1] ** 2
        zeta = alpha[1] * delta[0] / sigma
        eta = -alpha[1] * beta[1] / sigma

        Ay[0] = eta * Ay[0] + zeta * Ar[1]
        z = eta * z - zeta * Ar[0]
        Ar[0] -= Ay[0]
        Ar[1] = dot(A,Ar[0])
        x -= z

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

            sigma = alpha[2] * delta[0] - beta[1 ]** 2
            zeta = alpha[1] * delta[0] / sigma
            eta = -alpha[1] * beta[1] / sigma

            Ay[0] = eta * Ay[0] + zeta * Ar[1]
            z = eta * z - zeta * Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = dot(A,Ar[0])
            x -= z

        solution_updates[i + 1 - dif] = solution_updates[i - dif] + k + 1

    else:
        isConverged = False
    
    num_of_iter = i + 1
    residual_index = i - dif

    end(start_time, isConverged, num_of_iter, residual, residual_index, final_k = k)

    return isConverged