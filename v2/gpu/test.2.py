import numpy as np
import cupy as cp
from cupy.cuda import Device
from time import perf_counter

gpus = cp.cuda.runtime.getDeviceCount()
rate = 4
N = gpus * rate
local_N = N // gpus
T = np.float64
# shape = (N, N)
shape = (local_N, N)

results = [None] * 4
# b = cp.arange(N, dtype=T)
b = cp.zeros(N, dtype=T)

# init device
for i in range(gpus):
	with Device(i):
		pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
		cp.cuda.set_allocator(pool.malloc)

# check P2P
for peerDevice in range(gpus):
	for device in range(gpus):
		if peerDevice == device:
			continue
		flag = cp.cuda.runtime.deviceCanAccessPeer(device, peerDevice)
		print(
			f'Can access #{peerDevice} memory from #{device}: '
			f'{flag == 1}')

# calc
s = perf_counter()
for i in range(gpus):
	with Device(i):
		results[i] = b * i
print(perf_counter() - s)

#print 
for i in range(gpus):
	with Device(i):
		print(results[i])