# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:15:50 2017

@author: Andriy
"""

import numpy as np
import unittest
import time
from GradientDescent import  GradientDescent as grad

class GradientDescentTest(unittest.TestCase):
    
    def setUp(self):
        self.start = time.time()
    
    def tearDown(self):
        duration = time.time() - self.start
        print("{0}: {1:.4}".format(self.id(), duration))
    
    def testParabola(self):
        f = lambda x: x[0]**2
        opt = grad(f, tol = 1e-4, der_step=1e-5, desc_step = 0.05)
        res = opt.optimize(np.array([5]))
        expected = np.array([0])
        np.testing.assert_allclose(res,expected, atol=1e-3)
        print("Without moment: ",res)
        
        res = opt.optimize(np.array([-5]))
        np.testing.assert_allclose(res,expected, atol=1e-3)
        
    def testParabolaMomentum(self):
        f = lambda x: x[0]**2
        opt = grad(f, tol = 1e-4, der_step=1e-5, desc_step = 0.05, mom=0.2)
        res = opt.optimize(np.array([5]))
        expected = np.array([0])
        np.testing.assert_allclose(res,expected, atol=1e-3)
        print("With moment: ",res)
        
        res = opt.optimize(np.array([-5]))
        np.testing.assert_allclose(res,expected, atol=1e-3)
    
    def testRozenbrock(self):
        f = lambda x: 0.5*(1-x[0])**2 + (x[1]-x[0]**2)**2
        opt = grad(f, tol = 1e-4, der_step=1e-5, desc_step = 0.005)
        res = opt.optimize(np.array([5,5]))
        expected = np.array([1,1])
        np.testing.assert_allclose(res,expected, atol=1e-3)
        print("Without moment: ",res)
        
        res = opt.optimize(np.array([-5,-5]))
        np.testing.assert_allclose(res,expected, atol=1e-3)
        
    def testRozenbrockMomentum(self):
        f = lambda x: 0.5*(1-x[0])**2 + (x[1]-x[0]**2)**2
        opt = grad(f, tol = 1e-4, der_step=1e-5, desc_step = 0.005, mom=0.13)
        res = opt.optimize(np.array([5,5]))
        expected = np.array([1,1])
        np.testing.assert_allclose(res,expected, atol=1e-3)
        print("With moment: ",res)
        
        res = opt.optimize(np.array([-5,-5]))
        np.testing.assert_allclose(res,expected, atol=1e-3)
    
    def testParabolaGradient(self):
        points = np.array([[0],[1],[-1]])
        f = lambda x: x[0]**2
        opt = grad(f, der_step=1e-5)
        
        res = opt.calc_grad(points[0])
        expected = np.array([0])
        np.testing.assert_allclose(res,expected, atol=1e-3)
        
        res = opt.calc_grad(points[1])
        expected = np.array([2])
        np.testing.assert_allclose(res,expected, atol=1e-3)
        
        res = opt.calc_grad(points[2])
        expected = np.array([-2])
        np.testing.assert_allclose(res,expected, atol=1e-3)
    
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