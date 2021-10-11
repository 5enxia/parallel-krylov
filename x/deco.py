import numpy as np

def f(function):
    def g(*args, **kwargs):
        print('-')
        function(*args, **kwargs)
        print('-')
    return g


@f
def h():
	A = np.empty(10)
	b = np.empty(10)
	res = A.dot(b)
	print(res)

h()
