import numpy as np
import cupy as cp
from cupy.cuda import Device
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

# env
num_of_gpu = getDeviceCount()
print(num_of_gpu)

# load data
mpath =''
vpath =''
A = np.load(mpath)
b = np.load(vpath)

# padding data
old_N = b.size
num_of_append = num_of_process - (b.size % num_of_process) 
N = old_N + num_of_append
local_N = N // num_of_gpu
A = np.append(A, np.zeros((old_N, num_of_append)), axis=1)  # 右に0を追加
A = np.append(A, np.zeros((num_of_append, N)), axis=0)  # 下に0を追加
b = np.append(b, np.zeros(num_of_append))  # 0を追加

# time
###############################################################################
# scatter A
for i in range(num_of_gpu):
	with Device(i):
		begin = i*local_N
		end = (i + 1)*local_N
		local_A = cp.array(A[begin:end])

# scatter b & dot
res = np.zeros(b.size)
for i in range(num_of_gpu):
	with Device(i):
		begin = i*local_N
		end = (i + 1)*local_N
		local_b = cp.array(b[begin:end])
		res[begin:end] = local_A.dot(local_b).get()

print(res)
###############################################################################
