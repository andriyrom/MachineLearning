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
        clasters = []
        calc_again = True
        clasters_centers = self.__init_centers(count)
        while (calc_again):
            distances = self.__calc_distances(clasters_centers)
            clasters = self.__split_into_clasters(distances, clasters_centers)
            new_centers = self.__calc_new_centers(clasters)
            calc_again = self.__is_centers_far(clasters_centers, new_centers)
            clasters_centers = new_centers
        return clasters
    
    def __init_centers(self, count):
        points_count = self.__points.shape[0]
        indexes = np.random.choice(points_count, count, replace=False)
        return self.__points[indexes]
    
    #result: ndarray[claster, point] -> distance
    def __calc_distances(self, centers):
        centers_resh = centers.reshape((centers.shape[0],1,2))
        diffs = self.__points - centers_resh
        distances = np.apply_along_axis(np.linalg.norm, 2, diffs)
        return distances
    
    def __split_into_clasters(self, distances, clasters_centers):
        clasters_count = clasters_centers.shape[0]
        point_claster_indexes = np.argmin(distances,0)
        clasters = []
        for index in range(clasters_count):
            claster_points = self.__points[point_claster_indexes == index]
            clasters.append(Claster(clasters_centers[index], claster_points))
        return clasters
    
    def __calc_new_centers(self, clasters):
        num_clasters = len(clasters)
        new_centers = np.zeros((num_clasters, 2))
        for index in range(num_clasters):
            new_centers[index] = np.mean(clasters[index].points, axis=0)
        return new_centers
    
    def __is_centers_far(self, old_centers, new_centers):
        diffs = old_centers - new_centers
        distances = np.apply_along_axis(np.linalg.norm, 1, diffs)
        return distances.max() > 0.01
    
class Claster:
    #Center = ndarray["x", "y"]
    #Points = ndarray[["x", "y"],["x", "y"],...]   
    def __init__(self, center=None, points=None): 
        self.center = center
        self.points = points
    
    def __str__(self):
        center_str = str(self.center) 
        points_str = str(self.points)
        return "Center: {0} \nPoints: {1}".format(center_str, points_str)
    
    def __repr__(self):
        return str(self)