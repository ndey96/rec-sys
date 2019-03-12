# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:44:32 2019

@author: Matthew
"""
import matplotlib.pyplot as plt
from evaluation_hyperopt import get_metrics
import pandas as pd
from scipy.sparse import load_npz
from ALSpkNN import ALSpkNN
import numpy as np

import sys
sys.setrecursionlimit(10000)

def assign_defaults(default_vals, model):
    for key in list(default_vals.keys()):
        setattr(model, key, default_vals[key])

if __name__ == '__main__':
    
    load_data= False
    if load_data == True:
        print("Loading Data...")
        train_plays = load_npz('data/train_sparse.npz')
        test_plays = load_npz('data/test_sparse.npz')
        song_df = pd.read_hdf('data/song_df.h5', key='df')
        user_df = pd.read_hdf('data/user_df.h5', key='df')
        
        user_items_coo = test_plays.transpose().tocoo()
        user_to_listened_songs_map = {}
        print("Building user to listened song map...")
        for user_index, song_index in zip(user_items_coo.row, user_items_coo.col):
            if user_index not in user_to_listened_songs_map:
                user_to_listened_songs_map[user_index] = set()
            user_to_listened_songs_map[user_index].add(song_index)
    
    knn_frac_vals = [0,0.25,0.5,0.75]
    default_knn_frac = 0.5
    
    max_overlap_vals = [0.05,0.1,0.15,0.2]
    default_overlap = 0.1
    
    knn = [50,100,200,500]
    default_knn = 100
    
    #order of placement in dictionary matters
    varied_parameters = {}
    varied_parameters['k'] = knn
    varied_parameters['max_overlap'] = max_overlap_vals
    varied_parameters['knn_frac'] = knn_frac_vals

    default_vals = {}
    default_vals['k'] = default_knn
    default_vals['max_overlap'] = default_overlap
    default_vals['knn_frac'] = default_knn_frac
    
    metrics_to_get = ['MAP@K', 'mean_cosine_list_dissimilarity',"metadata_diversity"]
    
    xlabels = ['']
    
    print("Building model...")
    model = ALSpkNN(user_df, song_df, k=100, knn_frac=0.5, cf_weighting_alpha=1)
    print("Fitting model...")
    model.fit(train_plays)
    
    params_to_vary = list(varied_parameters.keys())
    
    USER_LIMIT = 50
    for param_name in params_to_vary:
        print("Starting:  " + str(param_name))
        results = np.zeros((len(metrics_to_get),len(varied_parameters[param_name])))
        for i,val in enumerate(varied_parameters[param_name]):
            setattr(model,param_name,val)
            metrics = get_metrics(
                    metrics=metrics_to_get,
                    N=20,
                    model=model,
                    train_user_items = train_plays.transpose(),
                    test_user_items = test_plays.transpose(),
                    song_df = song_df,
                    limit = USER_LIMIT,
                    user_to_listened_songs_map = user_to_listened_songs_map)
            results[:,i] = list(metrics.values())
        
        assign_defaults(default_vals,model)
        
        for i in range(len(metrics_to_get)):
            plt.figure()
            plt.title("Effect of varying " + param_name + " parameter")
            plt.plot(varied_parameters[param_name],results[i,:])
            plt.ylabel(metrics_to_get[i])
            plt.xlabel(param_name + " values")    
    