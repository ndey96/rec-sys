# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:56:23 2019

@author: Matthew
"""
import numpy as np
import random
from collections import Counter
from .utils import build_songs_listened_to_map


def filter_songs_func(song_count_tuple,song_blacklist):
    if song_count_tuple[0] in song_blacklist:
        return False
    else:
        return True

class WeightedRecommender:

    def recommend(self, user_sparse_index, train_plays_transpose=None, N=20):
        if user_sparse_index in self.user_to_listened_songs_map:
            songs_listened_to = self.user_to_listened_songs_map[
                user_sparse_index]
        else:
            songs_listened_to = []
            
        filtered_songs = filter(lambda x: filter_songs_func(x,songs_listened_to),self.song_counts)
        filtered_songs = list(filtered_songs)
        
        songs_possible = [song_tuple[0] for song_tuple in filtered_songs]
        songs_prob = [song_tuple[1] for song_tuple in filtered_songs]

        songs_prob = np.array(songs_prob)
        songs_prob = songs_prob / songs_prob.sum()

        recs = np.random.choice(
            a=songs_possible, p=songs_prob, size=N, replace=False)
        return recs

    def fit(self, train_csr):
        #csr matrix of item by users in same format as fit for CF model
        item_user = train_csr.tocoo()
        all_songs = item_user.row

        self.user_to_listened_songs_map = build_songs_listened_to_map(item_user)
        self.song_counts = Counter(all_songs).most_common()

