import sys
import unittest

sys.path.append('../../')
from krylov.util import loader, toepliz_matrix_generator
from krylov.method.single.adaptive_k_skip_mrr import adaptive_k_skip_mrr
import numpy as np 

class TestCgMethod(unittest.TestCase):
    epsilon = 1e-8
    T = np.float64

    def test_single_adaptive_k_skip_MrR_method(self):
        k = 10 
        A, b = toepliz_matrix_generator.generate()
        self.assertTrue(adaptive_k_skip_mrr(A, b, k, TestCgMethod.epsilon, TestCgMethod.T))

if __name__ == "__main__":
    unittest.main()