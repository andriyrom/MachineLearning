# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:35:48 2017

@author: Andriy
"""

import numpy as np

class GradientDescent:
    # function -> f(x), where x - numpy.array
    def __init__(self, function):
        self.__step = 0.02
        self.__descent_step = 0.01
        self.__func = function
    
    def optimize(self, start_point):
        point = start_point.astype("float64")
        gradient = self.calc_grad(point)
        new_step = True
        while (new_step):
            point -= self.__descent_step * gradient
            gradient = self.calc_grad(point)
            new_step = np.abs(gradient).max() > self.__descent_step
        return point
    
    def calc_grad(self, point):
        p_count = point.size
        new_points = np.eye(p_count) * self.__step + point
        new_points_values = np.apply_along_axis(self.__func, 1, new_points)
        partial_gains = new_points_values - self.__func(point)
        gradient = partial_gains / self.__step
        return gradient