# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 01:16:40 2019

@author: Matthew
"""
from random import shuffle
from copy import deepcopy

'''
Calculate the intersection over union for two lists as defined in:
    https://github.com/ndey96/rec-sys/issues/73
'''
def IOU(list_1,list_2):
    i = len(intersection(list_1,list_2))
    #union is the union of the two sets of numbers
    union = len(list_1)+len(list_2)-i
    return i/union    

def intersection(list_1,list_2):
    return set(list_1) & set(list_2)

'''
% of shared listens from list 1 and list 2 out of the total number of listens in list 1.
Alterative to IOU
'''
def shared_of_total(list_1,list_2):
    i = len(intersection(list_1,list_2))
    return i/len(list_1)
'''
Concatenate and shuffle two lists as defined in
 https://github.com/ndey96/rec-sys/issues/75
'''
def concat_shuffle(list_1,list_2):
    combined_list = deepcopy(list_1+list_2)
    shuffle(combined_list)
    return combined_list