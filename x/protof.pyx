import numpy as np
import cupy as cp
from mpi4py import MPI

def f(
	int k,
	cp.ndarray A,
	cp.ndarray x,
	cp.ndarray z,
	cp.ndarray Ar,
	cp.ndarray Ay,
	cp.ndarray alpha,
	cp.ndarray beta,
	cp.ndarray delta,
	np.ndarray Ar_cpu,
	):

	cdef int j, l
	cdef double zz, ee, ez, d, zeta, eta

	# MrRでの1反復(解と残差の更新)
	d = alpha[2] * delta[0] - beta[1] ** 2
	zeta = alpha[1] * delta[0] / d
	eta = -alpha[1] * beta[1] / d
	Ay[0] = eta * Ay[0] + zeta * Ar[1]
	z = eta * z - zeta * Ar[0]
	Ar[0] -= Ay[0]
	x -= z
	
	# MrRでのk反復
	for j in range(k):
		zz = zeta ** 2
		ee = eta ** 2
		ez = eta * zeta
		##
		delta[0] = zz * alpha[2] + ez * beta[1]
		alpha[0] -= zeta * alpha[1]
		##
		delta[1] = ee * delta[1] + 2 * eta * zeta * beta[2] + zz * alpha[3]
		beta[1] = eta * beta[1] + zeta * alpha[2] - delta[1]
		alpha[1] = -beta[1]
		for l in range(2, 2 * (k - j) + 1):
			##
			delta[l] = ee * delta[l] + 2 * ez * beta[l+1] + zz * alpha[l + 2]
			tau = eta * beta[l] + zeta * alpha[l + 1]
			##
			beta[l] = tau - delta[l]
			alpha[l] -= tau + beta[l]
		# 解と残差の更新
		d = alpha[2] * delta[0] - beta[1] ** 2
		##
		zeta = alpha[1] * delta[0] / d
		eta = -alpha[1] * beta[1] / d
		##
		MPI.COMM_WORLD.Allgather(A.dot(Ar[0]).get(), Ar_cpu[1])
		Ar[1] = cp.asarray(Ar_cpu[1])
		Ay[0] = eta * Ay[0] + zeta * Ar[1]
		z = eta * z - zeta * Ar[0]
		Ar[0] -= Ay[0]
		x -= z