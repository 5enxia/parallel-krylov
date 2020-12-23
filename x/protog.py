import cupy
from time import perf_counter


def g(comm, k, begin, end, A, x, z, Ar, Ay, alpha, beta, delta, Ar_cpu):
	dt = 0
	elapsed = 0

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

		dt = perf_counter()

		# comm.Allgather(A[begin:end].dot(Ar[0]), Ar[1])
		# comm.Allgather(A.dot(Ar[0]), Ar[1])
		# comm.Allgather(A[begin:end].dot(Ar[0]).get(), Ar_cpu[1])
		comm.Allgather(A.dot(Ar[0]).get(), Ar_cpu[1])
		Ar[1] = cupy.array(Ar_cpu[1])

		elapsed += perf_counter() - dt

		Ay[0] = eta * Ay[0] + zeta * Ar[1]
		z = eta * z - zeta * Ar[0]
		Ar[0] -= Ay[0]
		x -= z

	print(comm.Get_rank(), elapsed)
