import os
import sys
import datetime

import numpy as np

#from krylov.util.loader import load_condition_params
#epsilon, N, diag, sub_diag, k = load_condition_params('condition.json')
epsilon = 1e-8
#from krylov.util.toeplizmatrixgenerator import generate

#from krylov.method.processes.mrr import mrr
#from krylov.method.processes.kskipmrr import kskipmrr as kmrr
from krylov.method.processes.adaptivekskipmrr import adaptivekskipmrr as akmrr


T = np.float64
e = '.npy'
dd = 'data/Meshless-Matrix-Reduced'
pu = ['cpu', 'gpu'][1]

#t = ['EFG', 'reduced'][0]
#l = ['2801', '6881', '10601'][1]
#m = ['kskipmrr', 'adaptivekskipmrr'][1]
#nop = 16
k = int(sys.argv[1])
nop = int(sys.argv[2])
l = sys.argv[3]
#m = sys.argv[4]
#m = 'kskipmrr'
m = 'adaptivekskipmrr'
t = sys.argv[5]


A = np.load(f'{dd}/matrix_{t}-{l}{e}')
b = np.load(f'{dd}/vector_{t}-{l}{e}')
#a, b = generate(n, diag, sub_diag, t)


#t, nosl, residual = method(a, b, epsilon, t, pu)
#if m is 'kskipmrr':
    #time, nosl, residual = kmrr(A, b, epsilon, k, T, pu)
#time, nosl, residual, krylov_base_time = kmrr(A, b, epsilon, k, T, pu)
#else:
    #time, nosl, residual, k_history = akmrr(A, b, epsilon, k, T, pu)
time, nosl, residual, k_history, krylov_base_time = akmrr(A, b, epsilon, k, T, pu)

r = 'result'
now = datetime.datetime.now()
attempt = '02'
date = f'{now.year}{now.month:02}{now.day:02}_{attempt}'
dirs = f'{r}/{date}/{m}_{t}{l}_k{k:02}_{pu}{nop:02}'
try:
    os.makedirs(dirs)
except Exception as expt:
    pass

#np.save(f'{d}{c}/{n}/time{e}', t)
#np.save(f'{d}{c}/{n}/nosl{e}', nosl)
#np.save(f'{d}{c}/{n}/residual{e}', residual)

np.save(f'{dirs}/time{e}', time)
np.save(f'{dirs}/nosl{e}', nosl)
np.save(f'{dirs}/residual{e}', residual)
np.save(f'{dirs}/kbt{e}', krylov_base_time)
np.save(f'{dirs}/khistory{e}', k_history)

