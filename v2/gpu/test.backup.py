import numpy as np
import cupy as cp
from cupy.cuda import Device
from time import perf_counter

# params
num_iter = 1000
gpus = 4
N = 4
local_N = N // gpus
T = np.float64
shape = (N, N)

# matrix, vector
A_list = [None] * gpus
b_list = [None] * gpus
b = cp.ones(N, T)
y = cp.zeros(N, T)
y_list = [None] * gpus

def gen(i):
	arr = []
	for j in range(local_N):
		begin = (i*local_N+j)*N
		end = (i*local_N+j+1)*N
		arr.append(np.arange(begin, end, dtype=np.float64))
	return arr

# init device
for i in range(gpus):
	Device(i).use()
	pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
	cp.cuda.set_allocator(pool.malloc)

# P2P
for i in range(gpus):
	Device(i).use()
	for j in range(gpus):
		if i == j:
			continue
		flag = cp.cuda.runtime.deviceCanAccessPeer(i, j)
		res = 'Enable' if flag else 'Disable'
		print(f'{i} -> {j}: {res}')
		cp.cuda.runtime.deviceEnablePeerAccess(j)


# alloc
Device(0).use()
b = cp.ones(N, dtype=T)
for i in range(gpus):
	Device(i).use()
	A_list[i] = cp.array(gen(i), T)
	# A_list[i] = cp.ones(shape, T)
	A_list[i] = cp.arange(N*i, N*(i+1), dtype=T)
	# b_list[i] = cp.arange(N*i, N*(i+1), dtype=T)
	# y_list[i] = cp.zeros(N, T)
	y_list[i] = cp.zeros(local_N, T)

# dot
for i in range(gpus):
	Device(i).use()
	y_list[i] = A_list[i].dot(b)

# result
for i in range(gpus):
	Device(i).synchronize()
	print(y_list[i])



