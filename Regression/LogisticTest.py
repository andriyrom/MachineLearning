# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:58:16 2017

@author: Andriy
"""
import unittest
import numpy as np
from Logistic import Logistic, LogisticCustom

class LogisticTest(unittest.TestCase):
        
    def testOneParamLogistic(self):
        x_learn = np.array([[-5], [-4], [-1], [0],[3], [7], [4], [12]])
        y_learn = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lin_alg = Logistic(x_learn,y_learn)
        reg_func = lin_alg.regression()

        x_test = np.array([[-6], [-10], [-7], [-42],[-1.5], [-2]])
        y_test = np.array([0, 0, 0, 0, 0, 0])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-2)
        
        x_test = np.array([[6], [15], [21], [42],[3.3], [70]])
        y_test = np.array([1, 1, 1, 1, 1, 1])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-2)
        
    def testTwoParamLogistic(self):
        x_learn = np.array([[-5,7], [-4, 5], [-1, 12], [0, 11], [3, -1], [7, -10], [4, -6], [12, -7]])
        y_learn = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lin_alg = Logistic(x_learn,y_learn)
        reg_func = lin_alg.regression()
        
        x_test = np.array([[-6, 12], [-10, 10], [-7, 11], [-42, 21],[-1.5, 15], [-2, 23]])
        y_test = np.array([0, 0, 0, 0, 0, 0])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-2)
        
        x_test = np.array([[6, -5], [15, -5], [21, -12], [42, 0], [3.3, -7], [70, 2]])
        y_test = np.array([1, 1, 1, 1, 1, 1])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)
    
    def testOneParamLogisticCustom(self):
        x_learn = np.array([[-5], [-4], [-1], [0],[3], [7], [4], [12]])
        y_learn = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lin_alg = LogisticCustom(x_learn,y_learn)
        reg_func = lin_alg.regression()

        x_test = np.array([[-6], [-10], [-7], [-42],[-1.5], [-2]])
        y_test = np.array([0, 0, 0, 0, 0, 0])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-2)
        
        x_test = np.array([[6], [15], [21], [42],[3.3], [70]])
        y_test = np.array([1, 1, 1, 1, 1, 1])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-2)
        
    def testTwoParamLogisticCustom(self):
        x_learn = np.array([[-5,7], [-4, 5], [-1, 12], [0, 11], [3, -1], [7, -10], [4, -6], [12, -7]])
        y_learn = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lin_alg = LogisticCustom(x_learn,y_learn)
        reg_func = lin_alg.regression()
        
        x_test = np.array([[-6, 12], [-10, 10], [-7, 11], [-42, 21],[-1.5, 15], [-2, 23]])
        y_test = np.array([0, 0, 0, 0, 0, 0])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-2)
        
        x_test = np.array([[6, -5], [15, -5], [21, -12], [42, 0], [3.3, -7], [70, 2]])
        y_test = np.array([1, 1, 1, 1, 1, 1])
        y_res = reg_func(x_test)
        np.testing.assert_allclose(y_res,y_test, atol=1e-3)


if __name__ == "__main__":     
    suite = unittest.TestLoader().loadTestsFromTestCase(LogisticTest)
    unittest.TextTestRunner(verbosity=0).run(suite)