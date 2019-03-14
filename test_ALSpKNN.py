# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from ALSpkNN import ALSpkNN
from RandomRecommender import RandomRecommender
from implicit.evaluation import mean_average_precision_at_k

import time


#%%
load_data = False
if load_data == True:
    print("Loading Data")
    #    user_playlist_df = pd.read_hdf('data/userid_playlist.h5', key='df')
    #    user_MUSIC_df = pd.read_hdf('data/user_MUSIC_num_songs.h5', key='df')
    user_df = pd.read_hdf('data/user_df.h5', key='df')

    # train_plays, test_plays -> num_songs x num_users CSR matrix
    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')

#     # songs -> CSR_row_index: song_id
#     songs_mapping = pd.read_hdf('data/song_mapping.h5', key='df')

#     # users -> CSR_col_index: user_id
#     users_mapping = pd.read_hdf('data/user_mapping.h5', key='df')

    song_df = pd.read_hdf('data/song_df.h5', key='df')

#%%
#things that can be optimized:
# - The alpha value for confidence of the matrix factorization algorithm
# - The number of iterations in the als algorithm
# - the joining of n and m in the recommendation algirhtm


print("Building model...")
model = ALSpkNN(
    user_df,
    song_df,
    k=100,
    knn_frac=0.5,
    max_overlap=0.2,
    cf_weighting_alpha=1)
#model = RandomRecommender()
print("Fitting model...")
model.fit(train_plays)
recs = model.recommend(user_sparse_index=21, train_plays_transpose=train_plays.transpose(), N=5)

test_users = test_plays.tocoo().col[:1000]
num_recommendations = 20
print("Begin Testing...")
start = time.time()
for iteration,i in enumerate(test_users):
#    if i%250:
#        print("Starting iteration:  " + str(i))
    recs = model.recommend(user_sparse_index=i, train_plays_transpose=train_plays.transpose(), N=num_recommendations)
    
    if len(recs) != num_recommendations:
        print("UserId:  " + str(i) + " did not recieve enough recommendations")
        print("Number of Recommendations: " + str(len(recs)))
    
    if (len(list(set(recs))) != num_recommendations):
        print("UserId:  " + str(i) + " recieved duplicate recommendations")
        print("Number of Recommendations: " + str(len(recs)))

print("Testing took: " + str(time.time() - start) + "s")
    
