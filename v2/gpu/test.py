import numpy as np
import cupy as cp
from cupy.cuda import Device

gpus = 4
N = gpus * 2
local_N = N // gpus
T = np.float64
# shape = (N, N)
shape = (local_N, N)

A_list = [None] * 4
# b_list = [None] * 4
b = cp.arange(N, dtype=T)
y_list = [None] * 4

def gen(i):
	arr = []
	for j in range(local_N):
		begin = (i*local_N+j)*N
		end = (i*local_N+j+1)*N
		arr.append(np.arange(begin, end, dtype=np.float64))
	return arr


for i in range(4):
	Device(i).use()
	pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
	cp.cuda.set_allocator(pool.malloc)

	for j in range(gpus):
		if i == j:
			continue
		cp.cuda.runtime.deviceEnablePeerAccess(j)

	# A_list[i] = cp.ones(shape, T)
	# A_list[i] = cp.ones(N, T)

	A_list[i] = cp.array(gen(i), T)

	# b_list[i] = cp.arange(N*i, N*(i+1), dtype=T)
	y_list[i] = cp.zeros(N, T)

for i in range(4):
	for j in range(0, 4):
		cp.cuda.runtime.deviceCanAccessPeer(i, j)

for i in range(4):
	Device(i).use()
	y_list[i] = A_list[i].dot(b)
	Device(i).synchronize()
	print('A:', A_list[i].device, 'b', b.device, 'y', y_list[i].device, y_list[i])
