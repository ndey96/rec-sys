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

class ALSpkNN():
    '''
    knn_frac = % of KNN recommendations
    k = # of neighbours for KNN
    '''

    def __init__(self,
                 user_df,
                 user_mapping,
                 song_mapping,
                 k=100,
                 knn_frac=0.5,
                 cf_weighting_alpha=1):
        self.user_mapping = user_mapping
        self.song_mapping = song_mapping
        self.user_df = user_df
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
        weighted_train_csr = train_csr.copy()
        weighted_train_csr.data = 1 + np.log(
            self.cf_weighting_alpha * train_csr.data)
        self.cf_model.fit(weighted_train_csr)
    
    def calculate_overlap(self, list_1, list_2):
        overlap = len(set(list_1) & set(list_2))
        total = len(set(list_1)) + len(set(list_2))
        
        return float(overlap)/total
        
    def get_overlap_list(self, user_id, closest_user_songs):
        
        overlap_list = []
        songs = self.user_df.loc[self.user_df['user_id'] == user_id]['song_ids'].values[0]
        for i in range(len(closest_user_songs)):
            overlap_list.append(self.calculate_overlap(songs, closest_user_songs[i]))
        
        return overlap_list

    # Returns list of song_ids
    def get_knn_top_m_songs(self, user_id, m, min_overlap):

        user_MUSIC = self.user_df.loc[self.user_df['user_id'] == user_id][
            'MUSIC'].values[0]
        distances, indices = self.kdtree.query(user_MUSIC, self.k, p=1)
        # TODO: maybe sort closest_user_ids by distance if they are not already sorted?

        closest_user_ids = self.user_df.iloc[indices]['user_id'].tolist()

        # closest_user_songs -> list of lists of song_ids, len(closest_user_songs) == k
        closest_user_songs = self.user_df.loc[self.user_df['user_id'].isin(
            closest_user_ids)]['song_ids'].values
        
        # calculate overlap for all songlists and delete those without enough overlap
        insufficient_overlap_indices = []
        overlap_list = self.get_overlap_list(user_id, closest_user_songs)
        for i in range(len(closest_user_songs)):
            if overlap_list[i] < min_overlap:
                insufficient_overlap_indices.append(i)
        closest_user_songs = np.delete(closest_user_songs, insufficient_overlap_indices)

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

        # TODO: use a left join to speed this up?
        user_id = self.user_mapping.loc[self.user_mapping['sparse_index'] ==
                                        user_sparse_index]['user'].values[0]

        min_overlap = 0.05
        m_songs = self.get_knn_top_m_songs(user_id=user_id, m=m, min_overlap=min_overlap)

        # TODO: use a left join to speed this up?
        m_songs = [
            self.song_mapping.loc[self.song_mapping.track == song][
                'sparse_index'].values[0] for song in m_songs
        ]

        #I don't think score/confidence is used in MAP@k function, so it doesn't matter what value is filled
        hopefully_unimportant_val = 0.69

        m_songs = [(song, hopefully_unimportant_val) for song in m_songs]
        rec_list = utilities.concat_shuffle(n_songs, m_songs)
        return rec_list[:N]