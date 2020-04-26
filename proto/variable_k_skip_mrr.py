import sys
import unittest

sys.path.append('../../')
from krylov.util import loader
from krylov.single.variable_k_skip_mrr import variable_k_skip_mrr 
import numpy as np 

class TestCgMethod(unittest.TestCase):
    epsilon = 1e-8
    T = np.float64
    directory = '../data/'

    def test_single_variable_k_skip_MrR_method_with_EFG_1081_k_5(self):
        version, N, ext = 'EFG', 1081, '.txt'
        A = loader.matrixLoader(f'{TestCgMethod.directory}matrix_{version}-{N}{ext}', version, N, TestCgMethod.T)
        b = loader.vectorLoader(f'{TestCgMethod.directory}vector_{version}-{N}{ext}', version, N, TestCgMethod.T)
        k = 4 
        self.assertTrue(variable_k_skip_mrr(A, b, k, TestCgMethod.epsilon, TestCgMethod.T))

    def test_single_variable_k_skip_MrR_method_with_EFG_1081_k_7(self):
        version, N, ext = 'EFG', 1081, '.txt'
        A = loader.matrixLoader(f'{TestCgMethod.directory}matrix_{version}-{N}{ext}', version, N, TestCgMethod.T)
        b = loader.vectorLoader(f'{TestCgMethod.directory}vector_{version}-{N}{ext}', version, N, TestCgMethod.T)
        k = 7
        self.assertTrue(variable_k_skip_mrr(A, b, k, TestCgMethod.epsilon, TestCgMethod.T))

    def test_single_variable_k_skip_MrR_method_with_EFG_1081_k_10(self):
        version, N, ext = 'EFG', 1081, '.txt'
        A = loader.matrixLoader(f'{TestCgMethod.directory}matrix_{version}-{N}{ext}', version, N, TestCgMethod.T)
        b = loader.vectorLoader(f'{TestCgMethod.directory}vector_{version}-{N}{ext}', version, N, TestCgMethod.T)
        k = 10 
        self.assertTrue(variable_k_skip_mrr(A, b, k, TestCgMethod.epsilon, TestCgMethod.T))

if __name__ == "__main__":
    unittest.main()