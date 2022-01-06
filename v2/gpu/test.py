import numpy as np
import cupy as cp
from cupy.cuda import Device

N = 4
T = np.float64
shape = (N, N)
A_list = [None] * 4
# b_list = [None] * 4
b = cp.arange(N, dtype=T)
y_list = [None] * 4

for i in range(4):
	Device(i).use()
	pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
	cp.cuda.set_allocator(pool.malloc)

	cp.cuda.runtime.deviceEnablePeerAccess(i)

	A_list[i] = cp.ones(shape, T)
	# b_list[i] = cp.arange(N*i, N*(i+1), dtype=T)
	y_list[i] = cp.zeros(N, T)

for i in range(4):
	for j in range(0, 4):
		cp.cuda.runtime.deviceCanAccessPeer(i, j)

for i in range(4):
	Device(i).use()
	y_list[i] = A_list[i].dot(b)
	Device(i).synchronize()
	print(y_list[i])


