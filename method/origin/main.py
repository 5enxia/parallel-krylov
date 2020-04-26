from krylov import Methods,np
from loader import vectorLoader,matrixLoader

if __name__ == "__main__":
    epsilon = 1e-8
    T = np.float64
    directory = '../data/'

    version, N, ext = 'EFG', 1081, '.txt'
    A = matrixLoader(f'{directory}matrix_{version}-{N}{ext}', version, N, T)
    b = vectorLoader(f'{directory}vector_{version}-{N}{ext}', version, N, T)
    k = 7

    k_skip_mrr = Methods(A, b , T)
    k_skip_mrr.kskipmrr(k,T)