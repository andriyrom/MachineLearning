# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:51:40 2017

@author: Andriy
"""
import numpy as np

class KMeans:
    #Points = ndarray[["x", "y"],["x", "y"],...]
    def __init__(self, points):
        self.__points = points.copy()
    
    def get_clasters(self, count):
        return [Claster()]
    
class Claster:
    #Center = ndarray["x", "y"]
    #Points = ndarray[["x", "y"],["x", "y"],...]
    def __init__(self, center=None, points=None):
        self.center = center
        self.points = points
        
        