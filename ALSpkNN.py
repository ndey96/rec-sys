from implicit.als import AlternatingLeastSquares
import os
os.environ['MKL_NUM_THREADS'] = '1'
from scipy.spatial import KDTree
import numpy as np
from collections import Counter
import utilities
import time
from random import shuffle
import itertools
from itertools import filterfalse
import sys
import random
sys.setrecursionlimit(10000)


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
    k = # of neighbours for KNN
    knn_frac = % of KNN recommendations
    max_overlap = maximum % overlap between user and their MUSIC neighbours
    min_songs = only use users with > min_songs in our KNN code
    '''

    def __init__(self,
                 user_df,
                 song_df,
                 k=100,
                 knn_frac=0.5,
                 max_overlap=0.2,
                 cf_weighting_alpha=1,
                 min_songs=5):

        self.user_df = user_df
        self.song_df = song_df
        self.cf_weighting_alpha = cf_weighting_alpha
        self.knn_frac = knn_frac
        self.k = k
        self.max_overlap = max_overlap
        self.min_songs = min_songs

        user_df_subset = user_df.loc[user_df['num_songs'] > (min_songs - 1)]
        self.kdtree = KDTree(user_df_subset['MUSIC'].tolist())

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

    def calculate_overlap(self, list_1, list_2):
        overlap = len(set(list_1) & set(list_2))
        total = len(set(list_1)) + len(set(list_2))

        return float(overlap) / total

    def get_overlap_list(self, user_sparse_index,
                         closest_user_song_sparse_indices):

        overlap_list = []
        songs = self.user_df.loc[user_sparse_index]['song_sparse_indices']
        for i in range(len(closest_user_song_sparse_indices)):
            overlap_list.append(
                self.calculate_overlap(songs,
                                       closest_user_song_sparse_indices[i]))

        return overlap_list

    # Returns list of song_sparse_indices
    def get_knn_top_m_song_sparse_indices(self, user_sparse_index, m,
                                          max_overlap, songs_from_cf, mode):

        user_MUSIC = self.user_df.loc[user_sparse_index]['MUSIC']
        distances, indices = self.kdtree.query(user_MUSIC, self.k, p=1)
        # TODO: maybe sort closest_user_ids by distance if they are not already sorted?

        closest_user_song_sparse_indices = self.user_df.loc[indices][
            'song_sparse_indices'].values

        # calculate overlap for all songlists and delete those without enough overlap
        insufficient_overlap_indices = []

        overlap_list = self.get_overlap_list(user_sparse_index,
                                             closest_user_song_sparse_indices)
        for i in range(len(closest_user_song_sparse_indices)):
            if overlap_list[i] > max_overlap:
                insufficient_overlap_indices.append(i)

        #Users with only one or two songs in their listening history will almost
        # always exceed the overlap condition. This if statement checks if we
        # are clearing too many users. 5 was chosen as an arbitrary threshold
        if len(insufficient_overlap_indices) + 5 < len(
                closest_user_song_sparse_indices):
            closest_user_song_sparse_indices = np.delete(
                closest_user_song_sparse_indices, insufficient_overlap_indices)
        else:
            #Backup incase closest neighbours are all too similar to the user
            #Choose random MUSIC users since similarity of MUSIC scores has
            #became meaningless.
            random_sparse_user_indices = random.sample(
                list(self.user_df.index), m)
            closest_user_song_sparse_indices = self.user_df.loc[
                random_sparse_user_indices]['song_sparse_indices'].values

            print(
                "Choosing random users since not enough users have small enough overlap"
            )

        user_songs = self.user_df.loc[user_sparse_index]['song_sparse_indices']

        # closest_user_song_sparse_indices_flat -> list of song_ids
        closest_user_song_sparse_indices_flat = itertools.chain.from_iterable(
            closest_user_song_sparse_indices)

        filtered_songs = []
        for song in closest_user_song_sparse_indices_flat:
            if song not in (user_songs + songs_from_cf):
                filtered_songs.append(song)
                
        # m most popular songs are returned
        if mode == 'popular':
            top_m_songs = [i[0] for i in Counter(filtered_songs).most_common(m)]
            
        # random sample where more popular songs are weighted more heavily based on relative popularity
        elif mode == 'weighted_popular':
            top_m_songs = []
            
            while len(top_m_songs) < m:
                random.sample(filtered_songs, m - len(top_m_songs))
                top_m_songs = set(top_m_songs)
            
        # random sample where all songs are weighted equally regardless of popularity
        elif mode == 'random':
            top_m_songs = random.sample(set(filtered_songs), m)
            

        return top_m_songs

    # Returns [song_sparse_index]
    def recommend(self, user_sparse_index, train_plays_transpose, N):
        # m -> number of songs from KNN recs
        m = int(np.round(self.knn_frac * N))
        # n -> number of songs from CF recs
        n = N - m

        n_songs = []
        if n > 0:
            n_song_tuples = self.cf_model.recommend(
                userid=user_sparse_index, user_items=train_plays_transpose, N=n)
            n_songs = [song_tuple[0] for song_tuple in n_song_tuples]

        m_songs = []
        if m > 0:
            m_songs = self.get_knn_top_m_song_sparse_indices(
                user_sparse_index=user_sparse_index,
                m=m,
                max_overlap=self.max_overlap,
                songs_from_cf=n_songs,
                mode='random')

        rec_list = n_songs + m_songs
        # utilities.concat_shuffle(n_songs, m_songs)
        return rec_list[:N]
