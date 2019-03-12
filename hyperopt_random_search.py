from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist
from multiprocessing import Pool
import time
import pandas as pd
from scipy.sparse import load_npz
from ALSpkNN import get_baseline_cf_model, weight_cf_matrix, ALSpkNN
from functools import partial
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
import evaluation_hyperopt

N = 20

def log_uniform(min, max):
    return np.power(10, np.random.uniform(np.log10(min), np.log10(max)))

if __name__ == '__main__':

    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')
    song_df = pd.read_hdf('data/song_df.h5', key='df')
    user_df = pd.read_hdf('data/user_df.h5', key='df')

    ##################################################################

    # print("Building and fitting the baseline CF model")
    # baseline_cf_model = get_baseline_cf_model()
    # # weighted_train_csr = weight_cf_matrix(train_plays, alpha=1)
    # baseline_cf_model.fit(train_plays)

    # print("Evaluating the baseline CF model")
    # metrics = get_metrics(
    #     metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
    #     N=20,
    #     model=baseline_cf_model,
    #     train_user_items=train_plays.transpose(),
    #     test_user_items=test_plays.transpose(),
    #     song_df=song_df,
    #     limit=10000)
    # print(metrics)

    ##################################################################

    print("Building model...")
    model = ALSpkNN(user_df, song_df, k=100, knn_frac=0.25, cf_weighting_alpha=1, max_overlap=0.05)
    print("Fitting model...")
    model.fit(train_plays)

    limit_users = 10000
    
    with open('data/random_search_log.txt', 'w') as file:
        file.write('k, knn_frac, max_overlap, map_k, cosine, metadata\n')
    
    while True:
        
        model.k = int(log_uniform(10, 10000))
        model.knn_frac = random.uniform(0, 1)
        model.max_overlap = random.uniform(0, 0.5)
        
        print(f'Evaluating ALSpkNN with k={model.k}, knn_frac={model.knn_frac}, max_overlap={model.max_overlap}')

        metrics = evaluation_hyperopt.get_metrics(
            metrics=['MAP@K', 'mean_cosine_list_dissimilarity', 'metadata_diversity'],
            N=N,
            model=model,
            train_user_items=train_plays.transpose(),
            test_user_items=test_plays.transpose(),
            song_df=song_df,
            limit=limit_users)
        
        mapk = metrics["MAP@K"]
        cosdis = metrics["mean_cosine_list_dissimilarity"]
        metadata = metrics["metadata_diversity"]
        
        with open('data/random_search_log.txt', 'a') as file:
            file.write(f'{model.k},{model.knn_frac},{model.max_overlap},{mapk},{cosdis},{metadata}\n')

        

