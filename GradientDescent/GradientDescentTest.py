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
        f = lambda x: x**2
        opt = grad(f)
        res = opt.optimize(5)
        expected = 0
        np.testing.assert_allclose(res,expected, 1e-3)


unittest.main()        