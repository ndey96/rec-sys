from evaluation import get_metrics
from ALSpkNN import ALSpkNN
import pandas as pd

import numpy as np
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import matplotlib.pyplot as plt

from implicit.evaluation import mean_average_precision_at_k

# hyperparameters:
    #   k=100
    #   knn_frac=0.5
    #   min_overlap=0.05
    
k_vals = [5, 50, 500]
knn_frac_vals = [0.25, 0.5, 1]
min_overlap_vals = [0, 0.05, 0.2]

load_data = True
if load_data == True:
    print("Loading Data")
    user_df = pd.read_hdf('data/user_df.h5', key='df')

    # train_plays, test_plays -> num_songs x num_users CSR matrix
    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')

    song_df = pd.read_hdf('data/song_df.h5', key='df')
#     song_df.set_index('song_id', inplace=True)
    
with open('log.txt', 'w') as file:
    file.write('k, knn_frac, min_overlap, map_k, cosine\n')


for i in range(len(k_vals)):
    for j in range(len(knn_frac_vals)):
        for k in range(len(min_overlap_vals)):
            print(song_df.shape)
            tuning_model = ALSpkNN(user_df, song_df, k_vals[i], knn_frac_vals[j], min_overlap_vals[k], cf_weighting_alpha=1)
            print("Fitting model...")
            tuning_model.fit(train_plays)
            metrics = get_metrics(
                metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
                N=20,
                model=tuning_model,
                train_user_items=train_plays.transpose(),
                test_user_items=test_plays.transpose(),
                song_df=song_df,
                limit=10)
            
            mapk = metrics['MAP@K']
            cosdis = metrics['cosine_list_dissimilarity']
            
            with open('log.txt', 'a') as file:
                file.write(f'{k_vals[i]},{knn_frac_vals[j]},{min_overlap_vals[k]},{mapk},{cosdis}\n')