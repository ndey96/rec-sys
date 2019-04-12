# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 23:15:32 2019

@author: Matthew
"""

import os
os.environ['MKL_NUM_THREADS'] = '1'
from scipy.spatial import KDTree
import numpy as np
from collections import Counter
import time
import itertools
import sys
import random
sys.setrecursionlimit(10000)
import pandas as pd
from .ALS import ALSRecommender
from .PopularRecommender import PopularRecommender

class RandomUserRecommender():
    '''
    k = # of neighbours for KNN
    knn_frac = % of KNN recommendations
    mode = one of ['popular', 'weighted_random', 'random']
    '''

    def __init__(self,
                 user_df,
                 song_df,
                 k=150,
                 knn_frac=0.5,
                 max_overlap=0.2,
                 cf_weighting_alpha=1,
                 min_songs=5,
                 mode='popular'):

        self.user_df = user_df
        self.song_df = song_df
        self.cf_weighting_alpha = cf_weighting_alpha
        self.knn_frac = knn_frac
        self.k = k
        self.max_overlap = max_overlap
        self.min_songs = min_songs
        self.mode = mode

        als_params = {
            'factors': 16,
            'dtype': np.float32,
            'iterations': 2,
            'calculate_training_loss': True,
            'cf_weighting_alpha': 1
        }
        self.cf_model = ALSRecommender(**als_params)
        self.pop_model = PopularRecommender()

    def fit(self, train_csr):
#         self.cf_model.fit(train_csr)
        self.pop_model.fit(train_csr)

    def get_knn_top_m_song_sparse_indices(self, user_sparse_index, m,
                                          songs_from_cf):
        
        # Retrieve one extra user for the rare case that one of the sampled k users is the user itself        
        random_sparse_user_indices = random.sample(list(self.user_df.index), self.k+1)
        if user_sparse_index in random_sparse_user_indices:
            random_sparse_user_indices.remove(user_sparse_index)
        else:
            random_sparse_user_indices = random_sparse_user_indices[1:]
        
        random_user_song_sparse_indices = self.user_df.loc[
                random_sparse_user_indices]['song_sparse_indices'].values
        
        user_songs = self.user_df.loc[user_sparse_index]['song_sparse_indices']

        # closest_user_song_sparse_indices_flat -> list of song_ids
        random_user_song_sparse_indices_flat = itertools.chain.from_iterable(
            random_user_song_sparse_indices)

        filtered_songs = []
        for song in random_user_song_sparse_indices_flat:
            if song not in (user_songs + songs_from_cf):
                filtered_songs.append(song)

        # song_count_tuples -> format [(song_sparse_index, count)]
        song_count_tuples = Counter(filtered_songs).most_common()
        if len(song_count_tuples) < m:
            print('len(song_count_tuples) < m')

        top_songs = [song_tuple[0] for song_tuple in song_count_tuples]
        if self.mode == 'popular':
            m_songs = top_songs[:m]

        elif self.mode in ['weighted_random', 'random']:
            top_song_probs = None
            if self.mode == 'weighted_random':
                top_song_counts = [
                    song_tuple[1] for song_tuple in song_count_tuples
                ]
                top_song_probs = top_song_counts / np.sum(top_song_counts)

            m_song_count_tuples_indices = np.random.choice(
                len(song_count_tuples), p=top_song_probs, size=m, replace=False)
            m_song_count_tuples = [
                song_count_tuples[idx] for idx in m_song_count_tuples_indices
            ]
            # Although randomly sampled, the songs should still be sorted by popularity to maximize MAP@K
            m_song_count_tuples.sort(
                key=lambda song_tuple: song_tuple[1], reverse=True)

            m_songs = [song_tuple[0] for song_tuple in m_song_count_tuples]

        return m_songs

    # Returns [song_sparse_index]
    def recommend(self, user_sparse_index, train_plays_transpose, N):
        # m -> number of songs from KNN recs
        m = int(np.round(self.knn_frac * N))
        # n -> number of songs from CF recs
        n = N - m

        n_songs = []
        if n > 0:
#             n_song_tuples = self.cf_model.recommend(
#                 userid=user_sparse_index, user_items=train_plays_transpose, N=n)
#            n_songs = [song_tuple[0] for song_tuple in n_song_tuples]

            n_song_tuples = self.pop_model.recommend(
                user_sparse_index, train_plays_transpose, N=n)
            n_songs = n_song_tuples

        m_songs = []
        if m > 0:
            m_songs = self.get_knn_top_m_song_sparse_indices(
                user_sparse_index=user_sparse_index, m=m, songs_from_cf=n_songs)

        return n_songs + m_songs