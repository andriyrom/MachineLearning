# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:43:09 2017

@author: Andriy
"""

import unittest
import numpy as np
from Kmeans import KMeans

class KmeansTest(unittest.TestCase):
    
    def testClasterPartition(self):
        points = np.array([[1,2],[2,2], [3,1], [0,-5], [1,-7], [3, -6]])
        alg = KMeans(points)
        result = alg.get_clasters(1)
        
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].points, points)
        

unittest.main()