# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:58:14 2017

@author: Andriy
"""

import sys
if ".." not in sys.path:
    sys.path.append("..")
import numpy as np
from GradientDescent.GradientDescent import GradientDescent as grad

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Logistic:
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
        min_search = grad(self.__cost_func, tol=0.07, der_step = 1e-5, desc_step = 0.03, mom=0.2)
        params = min_search.optimize(init_vals)
        return self.__log_func(params)
    
    def __cost_func(self, params):
        sigm_x = sigmoid(self.__x.dot(params))
        cost_func_parts = -self.__y * np.log(sigm_x) - (1-self.__y)*np.log(1 - sigm_x)
        return cost_func_parts.sum()
    
    def __init_params(self, count):
        raw_vals = np.random.rand(count)
        return raw_vals/raw_vals.sum()
    
    def __log_func(self, params):
        def f(x):
            z = self.__extend_x(x).dot(params)
            return sigmoid(z)
        return f

class LogisticCustom:
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
        params = self.__minimize(init_vals, 0.07, 0.03, 0.2)
        return self.__log_func(params)
    
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
        diffs = sigmoid(self.__x.dot(params)) - self.__y
        grad = np.zeros(params.size)
        for index in range(params.size):
            column = self.__x[:,index]
            derivative = np.sum(column * diffs)
            grad[index] = derivative
        return grad
    
    def __init_params(self, count):
        raw_vals = np.random.rand(count)
        return raw_vals/raw_vals.sum()
    
    def __log_func(self, params):
        def f(x):
            z = self.__extend_x(x).dot(params)
            return sigmoid(z)
        return f