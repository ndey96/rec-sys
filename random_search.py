from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist
from multiprocessing import Pool
import time
import pandas as pd
from scipy.sparse import load_npz
from models.ALSpkNN import ALSpkNN
from functools import partial
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
from metrics import get_metrics


def log_uniform(min_val, max_val):
    return np.power(10, np.random.uniform(np.log10(min_val), np.log10(max_val)))

if __name__ == '__main__':

    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')
    song_df = pd.read_hdf('data/song_df.h5', key='df')
    user_df = pd.read_hdf('data/user_df.h5', key='df')

    start_time = int(time.time())
    # USER_LIMIT = 10000
    USER_LIMIT = 9999999
    N = 20
    file_path = f'random_searches/random_search_{start_time}.txt'
    with open(file_path, 'w') as file:
        file.write('k,knn_frac,max_overlap,mode,map_k,cosine,metadata\n')
    
    while True:
        try:
            print("Building model...")
            model = ALSpkNN(
                user_df,
                song_df,
                k=int(log_uniform(10, 10000)),
                knn_frac=random.uniform(0, 1),
                cf_weighting_alpha=1,
                max_overlap=random.uniform(0, 1),
                mode=random.choice(['popular', 'weighted_random', 'random']),
                min_songs=int(random.uniform(1, 15))
            )
            print("Fitting model...")
            model.fit(train_plays)
            
            print(f'Evaluating ALSpkNN with k={model.k}, knn_frac={model.knn_frac}, max_overlap={model.max_overlap}, min_songs={model.min_songs}')

            metrics = get_metrics(
                metrics=['MAP@K', 'mean_cosine_list_dissimilarity', 'metadata_diversity'],
                N=N,
                model=model,
                train_user_items=train_plays.transpose(),
                test_user_items=test_plays.transpose(),
                song_df=song_df,
                limit=USER_LIMIT)
            
            mapk = metrics["MAP@K"]
            cosdis = metrics["mean_cosine_list_dissimilarity"]
            metadata = metrics["metadata_diversity"]
            
            with open(file_path, 'a') as file:
                file.write(f'{model.k},{model.knn_frac},{model.max_overlap},{model.mode},{mapk},{cosdis},{metadata}\n')
        except Exception as e:
            print(f'Ran into the following exception: {e}')
        

