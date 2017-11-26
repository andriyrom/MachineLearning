# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:16:30 2017

@author: Andriy
"""
import numpy as np
from scipy.stats import norm

class NaiveBayes:
    #x_learn - numpy.array, dim = 2; y_learn - numpy.array, dim = 1;
    def __init__(self, x_learn, y_learn):
        self.__y_info = self.__yprob(y_learn)
        self.__x_info = self.__xprobs(x_learn)
        self.__xcondy_info = self.__xcondy_probs(x_learn, y_learn, self.__y_info.keys())
    
    def __yprob(self, y_learn):
        values, counts = np.unique(y_learn, return_counts = True) 
        probs = counts/counts.sum()
        return dict(zip(values, probs))
    
    def __xprobs(self, x_learn):
        probs = []
        features_count = x_learn.shape[1]
        for index in range(features_count):
            feature_vals = x_learn[:,index]
            prob = norm(feature_vals.mean(), feature_vals.std()).pdf
            probs.append(prob)
        return probs
    
    def __xcondy_probs(self, x_learn, y_learn, y_classes):
        result = {}
        for y_class in y_classes:
            xy_vals = x_learn[y_learn == y_class]
            cond_probs = self.__xprobs(xy_vals)
            result[y_class] = cond_probs
        return result
    
    #x - numpy.array, dim = 1
    def __classify_sample(self, x):
        posterior_probs = []
        x_joint_prob = self.__joint_prob(self.__x_info, x)
        classes = list(self.__y_info.keys())
        for y_class in classes:
            y_prob = self.__y_info[y_class]
            xcondy_joint_prob = self.__joint_prob(self.__xcondy_info[y_class], x)
            posterior_y = xcondy_joint_prob * y_prob / x_joint_prob
            posterior_probs.append(posterior_y)
        class_index = np.argmax(posterior_probs)
        return classes[class_index]
    
    #x - numpy.array, dim = 2
    def classify(self, x):
        return np.apply_along_axis(self.__classify_sample, 1, x)
    
    def __joint_prob(self, prob_funcs, x):
        features_cout = len(prob_funcs)
        x_probs = np.array([prob_funcs[i](x[i]) for i in range(features_cout)])
        return x_probs.prod()