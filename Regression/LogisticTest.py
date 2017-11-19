# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:58:16 2017

@author: Andriy
"""
import unittest
import numpy as np
from Logistic import Logistic

class LogisticTest(unittest.TestCase):
    def lin_func(self, x, params):
        samples_count = x.shape[0]
        extended_x = np.hstack((np.ones((samples_count, 1)), x))
        return extended_x.dot(params)
        
    def testOneParamLogistic(self):
        x_learn = np.array([[-5], [-4], [-1], [0],[3], [7], [4], [12]])
        y_learn = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lin_alg = Logistic(x_learn,y_learn)
        reg_func = lin_alg.regression()
        
        samples_count = 10
        x_test = np.array([[-6], [-10], [-7], [-42],[-1.5], [-2]])
        y_test = np.array([0, 0, 0, 0, 0, 0])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)
        
        x_test = np.array([[6], [15], [21], [42],[3.3], [70]])
        y_test = np.array([1, 1, 1, 1, 1, 1])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)


if __name__ == "__main__":     
    suite = unittest.TestLoader().loadTestsFromTestCase(LogisticTest)
    unittest.TextTestRunner(verbosity=0).run(suite)