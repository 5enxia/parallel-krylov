#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#------------------------------------------------------------------------------#
import sys
import os
import time
import datetime
import json

import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


class Methods():
    epsilon = 1e-10
    makersize=3
    lw = 1
    
    @staticmethod
    def multiplot(dataArray,figsize=(4,3),markersize=3):
        upper_limit = 0
        
        plt.figure(figsize=figsize)
        for d in dataArray:
            upper_limit = max(upper_limit, d.solution_updates[d.iter - 1])
        for d in dataArray:
            plt.plot(d.solution_updates[:d.iter], d.residual[:d.iter], '-o',
                     lw=1, label=d.name, markersize=markersize)

        plt.xlim(0,upper_limit+1)
        Methods._set_plot()
        plt.show()
        
    @staticmethod
    def _set_plot():
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["font.size"] = 32
        plt.yscale('log')
        plt.ylim(Methods.epsilon, 1)
        plt.grid()
        plt.xlabel('Number of Solution Updates')
        plt.ylabel('Residual Norm')
        plt.legend()
        plt.subplots_adjust(left=0.15, right=0.90, bottom=0.15, top=0.90)
    
    def __init__(self,A,b,T=np.float64):
        self.A = A
        self.b = b
        self.x = np.zeros(b.size,T)
        self.b_norm = np.linalg.norm(b)
        self.N = b.size
        self.converged = True
        self.max_iter = self.N * 2
        self.residual = np.zeros(self.max_iter ,T)
        self.solution_updates = np.zeros(self.max_iter+1,np.int)
        self.solution_updates[0] = 0

    def json2instance(self,fn):
        with open(fn,'r') as f:
            data = json.load(f)
            self.name = data['metadata']['method']
            self.solution_updates = data['results']['Number of Solution Updates']
            self.residual = data['results']['Residual Norm']
            self.iter = data['results']['iter']

    def plot(self,figsize=(4,3)):
        plt.figure(figsize=figsize)
        plt.plot(self.solution_updates[:self.iter],self.residual[:self.iter], 
                 '-o', lw=Methods.lw, label=self.name, markersize=Methods.makersize)
        Methods._set_plot()
        plt.show()
       
    def output(self,fn):
        # metadata
        metadata = dict()
        metadata['date'] = self.date
        metadata['time'] = self.time
        metadata['method'] = self.name
        metadata['epsilon'] = Methods.epsilon
        
        nosu = ("Number of Solution Updates", self.solution_updates[:self.iter].tolist())
        residual_norm = ('Residual Norm', self.residual[:self.iter].tolist())
        results = dict([
            ('converged',self.converged),('k',self.k),("iter",self.iter),nosu,residual_norm
        ])  
        
        output_data = {"metadata":metadata,'results':results}        
        
        with open(fn,'w') as f:
            json.dump(
                output_data, 
                f, 
                ensure_ascii=False, 
                indent=4, 
                sort_keys=False, 
                separators=(',', ': '))
            
    # ===============================Krylov Methods===============================
    def _setup(self,name,k=None):
        print('--------------------')
        self.name = name
        print(f'name:{self.name}')
        self.k = k
        self.date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.start = time.time()
    
    def _converged(self,iter_index,residual_index,k=None):
        print('Status: converged')
        print(f'iter: {iter_index} times')
        print(f'final_k: {k}')
        print(f'residual: {self.residual[residual_index]}')
        
    def _diverged(self):
        print('Status: Diverged')
        self.converged = False
        
    def _teardonw(self):
        self.time = time.time() - self.start
        print(f'time: {self.time}')
        print('--------------------')
        
    def cg(self,T=np.float64):
        self._setup(name='CG')
        
        r = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(r) / self.b_norm
        p = r.copy()

        for i in range(self.max_iter):
            alpha = r.dot(p) / p.dot(self.A).dot(p)
            self.x += alpha * p
            old_r = r.copy()
            r -= alpha * self.A.dot(p)

            self.residual[i+1] = np.linalg.norm(r) / self.b_norm
            if self.residual[i+1] < Methods.epsilon:
                self._converged(i,i+1)
                break

            beta = r.dot(r) / old_r.dot(old_r)
            p = r + beta * p
    
            self.solution_updates[i] = i
        
        else:
            self._diverged()
            
        self.iter = i
        self._teardonw()
        
    def pcg(self,M,T=np.float64):
        self._setup(name='Preconditioned CG')
        
        r = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(r) / self.b_norm
        z = M.dot(r)
        p = z.copy() 

        for i in range(self.max_iter):
            alpha = r.dot(z) / p.dot(self.A).dot(p)  
            self.x += alpha * p
            old_r = r.copy()
            old_z = z.copy()
            r -= alpha * self.A.dot(p)
            z = M.dot(r)

            self.residual[i+1] = np.linalg.norm(r) / self.b_norm
            if self.residual[i+1] < Methods.epsilon:
                self._converged(i,i+1)
                break

            beta = r.dot(z) / old_r.dot(old_z)
            p = z + beta * p
            
            self.solution_updates[i] = i
        
        else:
            self._diverged()
    
        self.iter = i
        self._teardonw()
        
    def mrr(self,T=np.float64):
        self._setup(name='MrR')
        
        r = np.zeros(self.max_iter, T)
        r = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(r) / self.b_norm
        z = np.zeros(self.N, T)

        Ar = self.A.dot(r)
        zeta = r.dot(Ar) / Ar.dot(Ar)
        y = zeta * Ar
        z = -zeta * r
        r -= y
        self.x -= z

        for i in range(1, self.max_iter):
            
            self.residual[i] = np.linalg.norm(r) / self.b_norm
            if self.residual[i] < Methods.epsilon:
                self._converged(i,i)
                break

            Ar = self.A.dot(r)
            nu = y.dot(Ar)
            gamma = nu / y.dot(y)
            s = Ar - gamma*y
            zeta = r.dot(s) / s.dot(s)
            # zeta = r.dot(Ar) / s.dot(s)
            eta = -zeta * gamma
            y = eta*y + zeta*Ar
            z = eta*z - zeta*r
            r -= y
            self.x -= z
            
            self.solution_updates[i] = i
        
        else:
            self._diverged()
        
        self.iter = i
        self._teardonw()
    
    def kskipcg(self,k,T=np.float64):
        self._setup('k-skip CG',k=k)
        
        Ar = np.zeros((k+2, self.N),T)
        Ar[0] = self.b - self.A.dot(self.x)
        Ap = np.zeros((k+3, self.N),T)
        Ap[0] = Ar[0]

        a = np.zeros(2*k+2, T)
        f = np.zeros(2*k+4, T)
        c = np.zeros(2*k+2, T)

        for i in range(0, self.max_iter):
            
            self.residual[i] = np.linalg.norm(Ar[0]) / self.b_norm
            if self.residual[i] < Methods.epsilon:
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
        self._teardonw()
      
    def kskipmrr(self,k,T=np.float64):
        self._setup('k-skip MrR',k=k)
        
        Ar = np.empty((k+3, self.N), T)
        Ar[0] = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(Ar[0]) / self.b_norm
        Ay = np.empty((k+2, self.N), T)

        #-----------
        # first iter
        Ar[1] = self.A.dot(Ar[0])
        zeta = Ar[0].dot(Ar[1]) / Ar[1].dot(Ar[1])
        Ay[0] = zeta * Ar[1]
        z = -zeta * Ar[0]
        Ar[0] -= Ay[0]
        self.x -= z
        #-----------

        alpha = np.empty(2*k+3, T)
        beta = np.empty(2*k+2, T)
        delta = np.empty(2*k+1, T)

        beta[0] = 0

        # Num of Solution Updates
        self.solution_updates[1] = 1

        for i in range(1, self.max_iter):

            self.residual[i] = np.linalg.norm(Ar[0]) / self.b_norm
            if self.residual[i] < Methods.epsilon:
                self._converged(i,i)
                break

            for j in range(1, k+2):
                Ar[j] = self.A.dot(Ar[j-1])

            for j in range(1, k+1):
                Ay[j] = self.A.dot(Ay[j-1])

            for j in range(2*k+3):
                jj = j // 2
                alpha[j] = Ar[jj].dot(Ar[jj + j%2])

            for j in range(1, 2*k+2):
                jj = j//2
                beta[j] = Ay[jj].dot(Ar[jj + j%2])

            for j in range(2*k+1):
                jj = j // 2
                delta[j] = Ay[jj].dot(Ay[jj + j%2])
                
            sigma = alpha[2]*delta[0] - beta[1]**2
            zeta = alpha[1]*delta[0] / sigma
            eta = -alpha[1]*beta[1] / sigma

            Ay[0] = eta*Ay[0] + zeta*Ar[1]
            z = eta*z - zeta*Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = self.A.dot(Ar[0])
            self.x -= z

            for j in range(k):

                delta[0] = zeta**2*alpha[2] + eta*zeta*beta[1]
                alpha[0] -= zeta*alpha[1]
                delta[1] = eta**2*delta[1] + 2*eta*zeta*beta[2] + zeta**2*alpha[3]
                beta[1] = eta*beta[1] + zeta*alpha[2] - delta[1]
                alpha[1] = -beta[1]

                for l in range(2, 2*(k-j)+1):

                    delta[l] = eta**2*delta[l] + 2*eta*zeta*beta[l+1] + zeta**2*alpha[l+2]
                    tau = eta*beta[l] + zeta*alpha[l+1]
                    beta[l] = tau - delta[l]
                    alpha[l] -= tau + beta[l]

                sigma = alpha[2]*delta[0] - beta[1]**2
                zeta = alpha[1]*delta[0] / sigma
                eta = -alpha[1]*beta[1] / sigma

                Ay[0] = eta*Ay[0] + zeta*Ar[1]
                z = eta*z - zeta*Ar[0]
                Ar[0] -= Ay[0]
                Ar[1] = self.A.dot(Ar[0])
                self.x -= z

            self.solution_updates[i + 1] = self.solution_updates[i] + k + 1

        else:
            self._diverged()
            
        self.iter = i + 1
        self._teardonw()

    def adaptivekskipmrr(self,k,T=np.float64):
        self._setup('adaptive k-skip MrR',k=k)
        
        #-----
        # init
        Ar = np.empty((k+3, self.N), T)
        Ar[0] = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(Ar[0]) / self.b_norm
        pre = self.residual[0]
        Ay = np.empty((k+2, self.N), T)
        #-----

        #-----------
        # first iter
        Ar[1] = self.A.dot(Ar[0])
        zeta = Ar[0].dot(Ar[1]) / Ar[1].dot(Ar[1])
        Ay[0] = zeta*Ar[1]
        z = -zeta*Ar[0]
        Ar[0] -= Ay[0]
        self.x -= z
        #-----------

        alpha = np.empty(2*k + 3, T)
        beta = np.empty(2*k + 2, T)
        delta = np.empty(2*k + 1, T)

        beta[0] = 0

        self.solution_updates[1] = 1

        dif = 0

        for i in range(1, self.max_iter):

            rrr = np.linalg.norm(Ar[0]) / self.b_norm

            if rrr > pre:
                self.x = pre_x.copy()
                Ar[0] = self.b - self.A.dot(self.x)
                Ar[1] = self.A.dot(Ar[0])
                zeta = Ar[0].dot(Ar[1]) / Ar[1].dot(Ar[1])
                Ay[0] = zeta * Ar[1]
                z = -zeta * Ar[0]
                Ar[0] -= Ay[0]
                self.x -= z

                if k > 1:
                    dif += 1
                    k -= 1

            else:
                pre = rrr
                self.residual[i - dif] = rrr
                pre_x = self.x.copy()

            if rrr < Methods.epsilon:
                self._converged(i,i-dif,k=k)
                break

            for j in range(1, k + 2):
                Ar[j] = self.A.dot(Ar[j-1])

            for j in range(1, k + 1):
                Ay[j] = self.A.dot(Ay[j-1])

            for j in range(2*k + 3):
                jj = j // 2
                alpha[j] = Ar[jj].dot(Ar[jj + j%2])

            for j in range(1, 2*k + 2):
                jj = j//2
                beta[j] = Ay[jj].dot(Ar[jj + j%2])

            for j in range(2*k + 1):
                jj = j // 2
                delta[j] = Ay[jj].dot(Ay[jj + j%2])

            sigma = alpha[2]*delta[0] - beta[1]**2
            zeta = alpha[1]*delta[0]/sigma
            eta = -alpha[1]*beta[1]/sigma

            Ay[0] = eta*Ay[0] + zeta*Ar[1]
            z = eta*z - zeta*Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = self.A.dot(Ar[0])
            self.x -= z

            for j in range(k):

                delta[0] = zeta**2*alpha[2] + eta*zeta*beta[1]
                alpha[0] -= zeta*alpha[1]
                delta[1] = eta**2*delta[1] + 2*eta*zeta*beta[2] + zeta**2*alpha[3]
                beta[1] = eta*beta[1] + zeta*alpha[2] - delta[1]
                alpha[1] = -beta[1]

                for l in range(2, 2*(k-j)+1):

                    delta[l] = eta**2*delta[l] + 2*eta*zeta*beta[l+1] + zeta**2*alpha[l+2]
                    tau = eta*beta[l] + zeta*alpha[l+1]
                    beta[l] = tau - delta[l]
                    alpha[l] -= tau + beta[l]

                sigma = alpha[2]*delta[0] - beta[1]**2
                zeta = alpha[1]*delta[0] / sigma
                eta = -alpha[1]*beta[1] / sigma

                Ay[0] = eta*Ay[0] + zeta*Ar[1]
                z = eta*z - zeta*Ar[0]
                Ar[0] -= Ay[0]
                Ar[1] = self.A.dot(Ar[0])
                self.x -= z

            self.solution_updates[i + 1 - dif] = self.solution_updates[i - dif] + k + 1

        else:
            self._diverged()
            
        self.iter = i + 1
        self._teardonw()
   
    def variablekskipmrr(self,k,T=np.float64):
        self._setup('variable k-skip MrR',k=k)
        
        tmp = k * 100
        #-----
        # init
        Ar = np.empty(((k+3) * tmp, self.N), T)
        Ar[0] = self.b - self.A.dot(self.x)
        self.residual[0] = np.linalg.norm(Ar[0]) / self.b_norm
        pre = self.residual[0]
        Ay = np.empty(((k+2) * tmp, self.N), T)
        #-----

        #-----------
        # first iter
        Ar[1] = self.A.dot(Ar[0])
        zeta = Ar[0].dot(Ar[1]) / Ar[1].dot(Ar[1])
        Ay[0] = zeta * Ar[1]
        z = -zeta * Ar[0]
        Ar[0] -= Ay[0]
        self.x -= z
        #-----------

        alpha = np.empty((2*k+3) * tmp, T) #modified
        beta = np.empty((2*k+2) *tmp, T) #modified
        delta = np.empty((2*k+1) * tmp, T) #modified

        beta[0] = 0

        self.solution_updates[1] = 1

        dif = 0
        count = 0

        for i in range(1, self.max_iter):

            rrr = np.linalg.norm(Ar[0]) / self.b_norm

            if rrr > pre:

                self.x = pre_x.copy()
                Ar[0] = self.b - self.A.dot(self.x)
                Ar[1] = self.A.dot(Ar[0])
                zeta = Ar[0].dot(Ar[1]) / Ar[1].dot(Ar[1])
                Ay[0] = zeta * Ar[1]
                z = -zeta * Ar[0]
                Ar[0] -= Ay[0]
                self.x -= z

                if k > 1:
                    dif += 1
                    k -= 1
                    if count > 1:
                        count -= 1

            else:
                pre = rrr
                self.residual[i - dif] = rrr
                pre_x = self.x.copy()
                
                # test
                count += 1
                if count > 1:
                    k += 1


            if rrr < Methods.epsilon:
                self._converged(i,i-dif,k)
                break

            for j in range(1, k+2):
                Ar[j] = self.A.dot(Ar[j-1])

            for j in range(1, k+1):
                Ay[j] = self.A.dot(Ay[j-1])

            for j in range(2*k+3):
                jj = j // 2
                alpha[j] = Ar[jj].dot(Ar[jj + j%2])

            for j in range(1, 2*k+2):
                jj = j//2
                beta[j] = Ay[jj].dot(Ar[jj + j%2])

            for j in range(2*k+1):
                jj = j // 2
                delta[j] = Ay[jj].dot(Ay[jj + j%2])

            sigma = alpha[2]*delta[0] - beta[1]**2
            zeta = alpha[1]*delta[0] / sigma
            eta = -alpha[1]*beta[1] / sigma

            Ay[0] = eta*Ay[0] + zeta*Ar[1]
            z = eta*z - zeta*Ar[0]
            Ar[0] -= Ay[0]
            Ar[1] = self.A.dot(Ar[0])
            self.x -= z

            for j in range(k):

                delta[0] = zeta**2*alpha[2] + eta*zeta*beta[1]
                alpha[0] -= zeta*alpha[1]
                delta[1] = eta**2*delta[1] + 2*eta*zeta*beta[2] + zeta**2*alpha[3]
                beta[1] = eta*beta[1] + zeta*alpha[2] - delta[1]
                alpha[1] = -beta[1]

                for l in range(2, 2*(k-j)+1):

                    delta[l] = eta**2*delta[l] + 2*eta*zeta*beta[l+1] + zeta**2*alpha[l+2]
                    tau = eta*beta[l] + zeta*alpha[l+1]
                    beta[l] = tau - delta[l]
                    alpha[l] -= tau + beta[l]

                sigma = alpha[2]*delta[0] - beta[1]**2
                zeta = alpha[1]*delta[0] / sigma
                eta = -alpha[1]*beta[1] / sigma

                Ay[0] = eta*Ay[0] + zeta*Ar[1]
                z = eta*z - zeta*Ar[0]
                Ar[0] -= Ay[0]
                Ar[1] = self.A.dot(Ar[0])
                self.x -= z

            self.solution_updates[i + 1 - dif] = self.solution_updates[i - dif] + k + 1

        else:
            self._diverged()
        
        self.iter = i + 1
        self._teardonw()
    # ===============================Krylov Methods===============================

