import random
from time import perf_counter

def f(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

arr = [
    10000,
    100000,
    1000000,
    10000000,
    100000000
]
for a in arr:
    s = perf_counter()
    f(a)
    print(perf_counter() - s)
