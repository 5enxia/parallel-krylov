import numpy as np
import cupy as cp
from cupy.cuda import Device
from time import perf_counter

gpus = 4
rate = 5000
N = gpus * rate
local_N = N // gpus
T = np.float64
# shape = (N, N)
shape = (local_N, N)

A_list = [None] * gpus
b_list = [None] * gpus
b = cp.arange(N, dtype=T)
y_list = [None] * gpus

streams = [None] * gpus

# arr gen
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
	streams[i] = cp.cuda.Stream()
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
		# print(f'{i} -> {j}: {res}', end='')
		cp.cuda.runtime.deviceEnablePeerAccess(j)
		# print(i, j)
		# print()


# alloc
Device(0).use()
b = cp.ones(N, dtype=T)
for i in range(gpus):
	Device(i).use()
	A_list[i] = cp.array(gen(i), T)
	b_list[i] = cp.ones(N, T)
	y_list[i] = cp.zeros(N, T)

dic = {'A': 'sync', 'B': 'async'}
print(dic['A'])
# dot
for k in range(5):
	s = perf_counter()
	for j in range(10000):
		for i in range(gpus):
			Device(i).use()
			cp.cuda.runtime.memcpyPeer(b_list[i].data.ptr, 0, b.data.ptr, i, b.nbytes)
			# cp.cuda.runtime.memcpyPeerAsync(b_list[i].data.ptr, 0, b.data.ptr, i, b.nbytes, streams[i].ptr)
			y_list[i] = A_list[i].dot(b_list[i])
		# result
		for i in range(gpus):
			# streams[i].synchronize()
			Device(i).synchronize()
	print(perf_counter() - s)