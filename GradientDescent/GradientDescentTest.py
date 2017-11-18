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
        np.testing.assert_allclose(res,expected, atol=1e-2)
        
        res = opt.optimize(np.array([-5]))
        np.testing.assert_allclose(res,expected, atol=2e-2)
    
    def testParabolaGradient(self):
        points = np.array([[0],[1],[-1]])
        f = lambda x: x[0]**2
        opt = grad(f)
        
        res = opt.calc_grad(points[0])
        expected = np.array([0.02])
        np.testing.assert_allclose(res,expected, atol=1e-2)
        
        res = opt.calc_grad(points[1])
        expected = np.array([2.02])
        np.testing.assert_allclose(res,expected, atol=1e-2)
        
        res = opt.calc_grad(points[2])
        expected = np.array([-1.98])
        np.testing.assert_allclose(res,expected, atol=1e-2)
    
    def testToleranceParameter(self):
        f = lambda x: x[0]**2
        opt = grad(f)
        expected = 0.01        
        self.assertEqual(opt.tolerance, expected)
        
        expected = 0.003
        opt = grad(f, expected)
        self.assertEqual(opt.tolerance, expected)
    
    def testDescent_StepParameter(self):
        f = lambda x: x[0]**2
        opt = grad(f)
        expected = 0.01        
        self.assertEqual(opt.descent_step, expected)
        
        expected = 0.003
        opt = grad(f, desc_step= expected)
        self.assertEqual(opt.descent_step, expected)
        
        expected = 0.003
        tol = 0.01
        opt = grad(f, tol, expected)
        self.assertEqual(opt.descent_step, expected)
    
    def testDerivative_StepParameter(self):
        f = lambda x: x[0]**2
        opt = grad(f)
        expected = 0.01        
        self.assertEqual(opt.derivative_step, expected)
        
        expected = 0.003
        opt = grad(f, der_step= expected)
        self.assertEqual(opt.derivative_step, expected)
        
        expected = 0.003
        tol = desc = 0.01
        opt = grad(f, tol, der_step= expected)
        self.assertEqual(opt.derivative_step, expected)
        opt = grad(f, tol, desc, expected)
        self.assertEqual(opt.derivative_step, expected)
    
    
unittest.main()        