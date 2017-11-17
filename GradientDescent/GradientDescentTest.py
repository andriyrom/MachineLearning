# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:15:50 2017

@author: Andriy
"""

import numpy as np
import unittest
from GradientDescent import  GradientDescent as grad

class GradientDescentTest(unittest.TestCase):
    
    def testParabola(self):
        f = lambda x: x[0]**2
        opt = grad(f)
        res = opt.optimize(np.array([5]))
        expected = np.array([0])
        np.testing.assert_allclose(res,expected, 1e-3)
    
    def testParabolaGradient(self):
        points = np.array([[0],[1],[-1]])
        f = lambda x: x[0]**2
        opt = grad(f)
        
        res = opt.calc_grad(points[0])
        expected = np.array([0.02])
        np.testing.assert_allclose(res,expected, 1e-2)
        
        res = opt.calc_grad(points[1])
        expected = np.array([2.02])
        np.testing.assert_allclose(res,expected, 1e-2)
        
        res = opt.calc_grad(points[2])
        expected = np.array([-1.98])
        np.testing.assert_allclose(res,expected, 1e-2)

unittest.main()        