import numpy as np
from krylov.util.loader import load_condition_params
from krylov.util.toeplizmatrixgenerator import generate

epsilon, N, diag, sub_diag, k = load_condition_params('condition.json')
A, b = generate(N, diag, sub_diag)

# 1
local_A1 = A[0:N//2]
local_A2 = A[N//2:N]
Ab = A.dot(b)
local_Ab1 = local_A1.dot(b)
local_Ab2 = local_A2.dot(b)
local_Ab = np.append(local_Ab1, local_Ab2)
print('Ab:',Ab)
print('Ab1:',local_Ab1)
print('Ab2:',local_Ab2)
print('Ab - local_Ab:', Ab - local_Ab)

# 2
Ab = A.dot(Ab)
local_Ab1 = local_A1.T.dot(local_Ab1)
local_Ab2 = local_A2.T.dot(local_Ab2)
local_Ab = local_Ab1 + local_Ab2
print('Ab:',Ab)
print('Ab1:',local_Ab1)
print('Ab2:',local_Ab2)
print('Ab - local_Ab:', Ab - local_Ab)

# 3
Ab = A.dot(Ab)
local_Ab1 = local_A1.dot(local_Ab1)
local_Ab2 = local_A2.dot(local_Ab2)
print('Ab:',Ab)
print('Ab1:',local_Ab1)
print('Ab2:',local_Ab2)