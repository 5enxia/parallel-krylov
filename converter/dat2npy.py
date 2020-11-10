import numpy as np

fileName = 'mat.dat'
with open(f'../data/{fileName}') as f:
    M, N, L = map(int, f.readline().split())
    arr = np.zeros((M, N))
    line = f.readline().split()
    for i in range(L):
        line = f.readline().split()
        arr[int(line[0]), int(line[1])] = float(line[2])
    np.save(f'../data/{fileName}.npy', arr)

# fileName = 'bvec.dat'
# with open(f'../data/{fileName}') as f:
#     arr = []
#     N = int(f.readline())
#     for i in range(N):
#         arr.append(float(f.readline()))
#     np.save(f'../data/{fileName}.npy', arr)
