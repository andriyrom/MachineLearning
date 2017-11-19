# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:41:14 2017

@author: Andriy
"""

import unittest
import numpy as np
from Linear import Linear, LinearCustom

class LinearTest(unittest.TestCase):
    def lin_func(self, x, params):
        samples_count = x.shape[0]
        extended_x = np.hstack((np.ones((samples_count, 1)), x))
        return extended_x.dot(params)
        
    def testOneParamRegression(self):
        params = np.array([1, -0.5])
        samples_count = 5
        x_learn = np.linspace(-10, 10, samples_count).reshape((samples_count, 1))
        y_learn = self.lin_func(x_learn, params)
        lin_alg = Linear(x_learn,y_learn)
        reg_func = lin_alg.regression()
        
        samples_count = 10
        x_test = np.linspace(20, 27, samples_count).reshape((samples_count, 1))
        y_test = self.lin_func(x_test, params)
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)
        
        x_test = np.linspace(-42, -35, samples_count).reshape((samples_count, 1))
        y_test = self.lin_func(x_test, params)
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)
    
    def testOneParamCustomRegression(self):
        params = np.array([1, -0.5])
        samples_count = 5
        x_learn = np.linspace(-10, 10, samples_count).reshape((samples_count, 1))
        y_learn = self.lin_func(x_learn, params)
        lin_alg = LinearCustom(x_learn,y_learn)
        reg_func = lin_alg.regression()
        
        samples_count = 10
        x_test = np.linspace(20, 27, samples_count).reshape((samples_count, 1))
        y_test = self.lin_func(x_test, params)
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)
        
        x_test = np.linspace(-42, -35, samples_count).reshape((samples_count, 1))
        y_test = self.lin_func(x_test, params)
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)

if __name__ == "__main__":     
    suite = unittest.TestLoader().loadTestsFromTestCase(LinearTest)
    unittest.TextTestRunner(verbosity=0).run(suite)