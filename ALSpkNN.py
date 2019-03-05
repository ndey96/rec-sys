from implicit.als import AlternatingLeastSquares
import os
os.environ['MKL_NUM_THREADS'] = '1'
from scipy.spatial import KDTree
import numpy as np
from collections import Counter
import utilities
import time
from random import shuffle


def get_baseline_cf_model():
    als_params = {
        'factors': 16,
        'dtype': np.float32,
        'iterations': 2,
        'calculate_training_loss': True
    }
    cf_model = AlternatingLeastSquares(**als_params)
    return cf_model


def weight_cf_matrix(csr_mat, alpha):
    #don't want to modify original incase it gets put into other models
    weighted_csr_mat = csr_mat.copy()
    weighted_csr_mat.data = 1 + np.log(alpha * csr_mat.data)
    return weighted_csr_mat


class ALSpkNN():
    '''
    knn_frac = % of KNN recommendations
    k = # of neighbours for KNN
    '''

    def __init__(self,
                 user_df,
                 song_df,
                 k=100,
                 knn_frac=0.5,
                 cf_weighting_alpha=1):
        self.user_df = user_df
        self.song_df = song_df
        self.cf_weighting_alpha = cf_weighting_alpha
        self.knn_frac = knn_frac
        self.k = k
        self.kdtree = KDTree(user_df['MUSIC'].tolist())

        #build the collaborative filtering model with params hardcoded
        als_params = {
            'factors': 16,
            'dtype': np.float32,
            'iterations': 2,
            'calculate_training_loss': True
        }
        self.cf_model = AlternatingLeastSquares(**als_params)

    def fit(self, train_csr):
        #don't want to modify original incase it gets put into other models
        weighted_train_csr = weight_cf_matrix(train_csr,
                                              self.cf_weighting_alpha)
        self.cf_model.fit(weighted_train_csr)

    # Returns list of song_ids
    def get_knn_top_m_song_ids(self, user_id, m):
        user_MUSIC = self.user_df.loc[self.user_df['user_id'] == user_id][
            'MUSIC'].values[0]
        distances, indices = self.kdtree.query(user_MUSIC, self.k, p=1)
        # TODO: maybe sort closest_user_ids by distance if they are not already sorted?

        closest_user_ids = self.user_df.iloc[indices]['user_id'].to_list()

        # closest_user_songs -> list of lists of song_ids, len(closest_user_songs) == k
        closest_user_songs = self.user_df.loc[self.user_df['user_id'].isin(
            closest_user_ids)]['song_ids'].values

        # closest_user_songs_flat -> list of song_ids
        closest_user_songs_flat = itertools.chain.from_iterable(
            closest_user_songs)

        top_m_songs = [
            i[0] for i in Counter(closest_user_songs_flat).most_common(m)
        ]
        return top_m_songs

    # Returns [(song_sparse_index, confidence)]
    def recommend(self, user_sparse_index, train_plays, N):
        # m -> number of songs from KNN recs
        m = int(np.round(self.knn_frac * N))
        # n -> number of songs from CF recs
        n = N - m

        n_songs = self.cf_model.recommend(
            userid=user_sparse_index, user_items=train_plays.transpose(), N=n)

        user_id = self.user_df.loc[user_sparse_index]['user_id']
        m_song_ids = self.get_knn_top_m_song_ids(user_id=user_id, m=m)
        m_songs = self.song_df.loc[m_song_ids]['sparse_index'].tolist()

        #I don't think score/confidence is used in MAP@k function, so it doesn't matter what value is filled
        hopefully_unimportant_val = 0.69

        m_songs = [(song, hopefully_unimportant_val) for song in m_songs]
        rec_list = utilities.concat_shuffle(n_songs, m_songs)
        return rec_list[:N]