# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:38:44 2017

@author: Andriy
"""
import unittest
import numpy as np
from knn import knn

class KnnTest(unittest.TestCase):
    def testOneClass(self):
        start = 9
        count = 7
        x_learn = np.random.normal(start, 2, size = (count, 2))
        y_learn = np.zeros(count)
        alg = knn(x_learn, y_learn) 
        
        neighbors = 3
        test_x = [7,7]
        test_y = 0
        res_y = alg.classify(test_x, neighbors)
        np.testing.assert_equal(res_y, test_y)
        
        test_x = [-3,-3]
        test_y = 0
        res_y = alg.classify(test_x, neighbors)
        np.testing.assert_equal(res_y, test_y)
        
        
    def testTwoClasses(self):
        start = 9
        count = 7
        claster1 = np.random.normal(start, 2, size = (count, 2))
        clas1 = np.zeros(count)
        start = -5
        claster2 = np.random.normal(start, 2, size = (count, 2))
        clas2 = np.ones(count)
        x_learn = np.vstack((claster1, claster2))
        y_learn = np.hstack((clas1, clas2))
        alg = knn(x_learn, y_learn) 
        
        neighbors = 3
        test_x = [7,7]
        test_y = 0
        res_y = alg.classify(test_x, neighbors)
        np.testing.assert_equal(res_y, test_y)
        
        test_x = [-3,-3]
        test_y = 1
        res_y = alg.classify(test_x, neighbors)
        np.testing.assert_equal(res_y, test_y)
        
        neighbors = 5
        test_x = [1,1]
        test_y = 1
        res_y = alg.classify(test_x, neighbors)
        np.testing.assert_equal(res_y, test_y)

if __name__ == "__main__":
    tests = unittest.TestLoader().loadTestsFromTestCase(KnnTest)
    unittest.TextTestRunner(verbosity=2).run(tests)