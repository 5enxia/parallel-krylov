#!/usr/bin/env python
# coding: utf-8

# ================================== MODULE ================================== #
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
# ============================================================================ #

class Krylov():
    epsilon =  1e-10

    # ================================ COMMON ================================ #
    def __init__(self, A, b, T = np.float64):
        self.A = A
        self.b = b
        self.x = np.zeros(b.size,T)
        self.b_norm = np.linalg.norm(b)
        self.N = b.size
        self.max_iter = self.N # * 2 # must tuning
        self.residual = np.zeros(self.max_iter + 1,T)
        self.solution_updates = np.zeros(self.max_iter + 1, np.int)
        self.solution_updates[0] = 0

    def _start(self, name = '', k = None):
        print('# ============== INFO ================= #')
        print(f'Method:\t{ name }')
        print(f'k:\t{ k }')

        self.start = time.perf_counter()

    def _end(self):
        self.time = time.perf_counter() - self.start

        print(f'time:\t{ self.time } s')
        status = 'converged' if self.converged else 'diverged'
        print(f'status:\t{ status }')
        if self.converged:
            print(f'iter:\t{ self.iter_index } times')
            print(f'residual:\t{self.residual[self.residual_index]}')
        print('# ===================================== #')
    
    def _converged(self, iter_index, residual_index):
        self.converged = True
        self.iter_index = iter_index
        self.residual_index = residual_index
        
    def _diverged(self):
        self.converged = False
    # ======================================================================== #    

    # ================================== CG ================================== #
    def cg(self, T = np.float64):
        self._start(name = sys._getframe().f_code.co_name)
        
        r = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(r) / self.b_norm
        p = r.copy()

        for i in range(self.max_iter):
            alpha = r.dot(p) / p.dot(self.A).dot(p)
            self.x += alpha * p
            old_r = r.copy()
            r -= alpha * self.A.dot(p)

            self.residual[i+1] = np.linalg.norm(r) / self.b_norm
            if self.residual[i+1] < Krylov.epsilon:
                self._converged(i,i+1)
                break

            beta = r.dot(r) / old_r.dot(old_r)
            p = r + beta * p
    
            self.solution_updates[i] = i
        else:
            self._diverged()
            
        self.iter = i
        self._end()
    # ======================================================================== #    

    # ============================== K-SKIP CG =============================== #
    def k_skip_cg(self, k, T = np.float64):
        self._start(name = sys._getframe().f_code.co_name, k = k)
        
        Ar = np.zeros((k+2, self.N),T)
        Ar[0] = self.b - self.A.dot(self.x)
        Ap = np.zeros((k+3, self.N),T)
        Ap[0] = Ar[0]

        a = np.zeros(2*k+2, T)
        f = np.zeros(2*k+4, T)
        c = np.zeros(2*k+2, T)

        for i in range(0, self.max_iter):
            
            self.residual[i] = np.linalg.norm(Ar[0]) / self.b_norm
            if self.residual[i] < Krylov.epsilon:
                self._converged(i,i)
                break

            for j in range(1, k+1):
                Ar[j] = self.A.dot(Ar[j-1])

            for j in range(1, k+2):
                Ap[j] = self.A.dot(Ap[j-1])

            for j in range(0, 2*k+1, 2):
                jj = j // 2
                a[j] = Ar[jj].dot(Ar[jj])
                a[j+1] = Ar[jj].dot(Ar[jj+1])

            for j in range(0, 2*k+3, 2):
                jj = j // 2
                f[j] = Ap[jj].dot(Ap[jj])
                f[j+1] = Ap[jj].dot(Ap[jj+1])

            for j in range(0, 2*k+1, 2):
                jj = j // 2
                c[j] = Ar[jj].dot(Ap[jj])
                c[j+1] = Ar[jj].dot(Ap[jj+1])
    
            alpha = a[0] / f[1]
            beta = alpha**2 * f[2] / a[0] - 1
            self.x += alpha*Ap[0]
            Ar[0] -= alpha*Ap[1]
            Ap[0] = Ar[0] + beta*Ap[0]
            Ap[1] = self.A.dot(Ap[0])
        
            for j in range(k):

                for l in range(0, 2*(k-j)+1):
            
                    a[l] += alpha*(alpha*f[l+2] - 2*c[l+1])
                    d = c[l] - alpha*f[l+1]
                    c[l] = a[l] + d*beta
                    f[l] = c[l] + beta*(d + beta*f[l])
        
                alpha = a[0] / f[1]
                beta = alpha**2 * f[2] / a[0] - 1
                self.x += alpha*Ap[0]
                Ar[0] -= alpha*Ap[1]
                Ap[0] = Ar[0] + beta*Ap[0]
                Ap[1] = self.A.dot(Ap[0])
            
            self.solution_updates[i+1] = self.solution_updates[i] + k + 1

        else:
            self._diverged()
    
        self.iter = i + 1
        self._end()
    # ======================================================================== #    
