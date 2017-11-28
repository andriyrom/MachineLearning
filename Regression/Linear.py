# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:41:14 2017

@author: Andriy
"""

import sys
if ".." not in sys.path:
    sys.path.append("..")
import numpy as np
from GradientDescent.GradientDescent import GradientDescent as grad

class Linear:
    #x_vals - numpy.array, dim = 2; y_vals - numpy.array, dim=1 
    def __init__(self, x_vals, y_vals):
        self.__x = self.__extend_x(x_vals)
        self.__y = y_vals.copy()
    
    def __extend_x(self, x_vals):
        ones_arr = np.ones((x_vals.shape[0], 1))
        return np.hstack((ones_arr, x_vals))
    
    def regression(self):
        params_count = self.__x.shape[1]
        init_vals = self.__init_params(params_count)
        min_search = grad(self.__cost_func, tol=1e-3, der_step = 1e-5, desc_step = 0.001, mom=0.2)
        params = min_search.optimize(init_vals)
        return self.__lin_func(params)
    
    def __cost_func(self, params):
        expected = self.__x.dot(params)
        diff = expected-self.__y
        return 0.5*(diff**2).sum()
    
    def __init_params(self, count):
        raw_vals = np.random.rand(count)
        return raw_vals/raw_vals.sum()
    
    def __lin_func(self, params):
        def f(x):
            x_ext = self.__extend_x(x)
            return x_ext.dot(params)
        return f
    
class LinearCustom:
    #x_vals - numpy.array, dim = 2; y_vals - numpy.array, dim=1 
    def __init__(self, x_vals, y_vals):
        self.__x = self.__extend_x(x_vals)
        self.__y = y_vals.copy()
    
    def __extend_x(self, x_vals):
        ones_arr = np.ones((x_vals.shape[0], 1))
        return np.hstack((ones_arr, x_vals))
    
    def regression(self):
        params_count = self.__x.shape[1]
        init_vals = self.__init_params(params_count)
        params = self.__minimize(init_vals, 1e-3, 0.007, 0.09)
        return self.__lin_func(params)
    
    def __minimize(self, init_vals, tolerance, learn_rate, mom_rate):
        point = init_vals.astype("float64")
        gradient = self.__gradient(point)
        momentum = gradient * mom_rate
        new_step = True
        while (new_step):
            momentum = momentum * mom_rate + learn_rate * gradient
            point -= momentum
            gradient = self.__gradient(point)
            new_step = np.abs(gradient).max() > tolerance
        return point
    
    def __gradient(self, params):
        diffs = self.__x.dot(params) - self.__y
        grad = np.zeros(params.size)
        for index in range(params.size):
            column = self.__x[:,index]
            derivative = np.sum(column * diffs)
            grad[index] = derivative
        return grad
    
    def __init_params(self, count):
        raw_vals = np.random.rand(count)
        return raw_vals/raw_vals.sum()
    
    def __lin_func(self, params):
        def f(x):
            x_ext = self.__extend_x(x)
            return x_ext.dot(params)
        return f