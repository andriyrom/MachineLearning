# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:37:33 2017

@author: Andriy
"""
import numpy as np

class knn:
    #x_train - numpy.array, dim=2; y_train - numpy.array, dim=1;
    def __init__(self, x_train, y_train):
        self.__x = x_train.copy()
        self.__y = y_train.copy()
    
    def classify(self, point, neighbors_count):
        return -1