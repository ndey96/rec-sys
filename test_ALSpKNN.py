# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from ALSpkNN import ALSpkNN
from implicit.evaluation import mean_average_precision_at_k


#%%
class RandomRecommender:

    def __init__(self):
        #        self.min_bound = min_bound
        #        self.high_bound = high_bound
        #        self.confidence = confidence
        self.something = 1

    def recommend(self, userid, train_csr, N):
        recs = np.random.randint(self.min_bound, self.high_bound, N)
        recs = [(rec, self.confidence) for rec in recs]
        return recs

    def fit(self, train_data):
        self.min_bound = 0
        self.high_bound = train_data.shape[0]
        self.confidence = 1
        return 1


#%%
def baseline_cf_model(train_plays):
    als_params = {
        'factors': 16,
        'dtype': np.float32,
        'iterations': 2,
        'calculate_training_loss': True
    }
    cf_model = AlternatingLeastSquares(**als_params)
    return cf_model


#%%
load_data = True
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
    min_overlap=0.05,
    cf_weighting_alpha=1)
print("Fitting model...")
model.fit(train_plays)
recs = model.recommend(user_sparse_index=21, train_plays_transpose=train_plays, N=5)
print(recs)