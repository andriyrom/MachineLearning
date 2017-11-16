# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:43:09 2017

@author: Andriy
"""

import unittest
import numpy as np
from Kmeans import KMeans

class KmeansTest(unittest.TestCase):
    
    def testOneClaster(self):
        points = np.array([[1,2],[2,2], [3,1], [0,-5], [1,-7], [2, -6]])
        alg = KMeans(points)
        result = alg.get_clasters(1)
        
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].points, points)
        
    def testTwoClasters(self):
        claster1 = np.array([[1,2],[2,2], [3,1]]) 
        claster2 = np.array([[0,-5], [1,-7], [-1, -6]])
        all_points = np.vstack((claster1, claster2))
        alg = KMeans(all_points)
        result = alg.get_clasters(2)
        max_clas = 0 if result[0].center[1] > result[1].center[1] else 1
        min_clas = 1 - max_clas
        
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[max_clas].points, claster1, err_msg = "Expected:{0} \nActual: {1}\n".format(claster1, result[max_clas].points))
        np.testing.assert_allclose(result[min_clas].points, claster2, err_msg = "Expected:{0} \nActual: {1}\n".format(claster2, result[min_clas].points))
        

unittest.main()