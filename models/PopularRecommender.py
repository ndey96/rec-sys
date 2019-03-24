# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:56:23 2019

@author: Matthew
"""
import numpy as np
import random
from collections import Counter
from .utils import build_songs_listened_to_map


class PopularRecommender:

    def recommend(self, user_sparse_index, train_plays_transpose=None, N=20):
        if user_sparse_index in self.user_to_listened_songs_map:
            songs_listened_to = self.user_to_listened_songs_map[
                user_sparse_index]
        else:
            songs_listened_to = set()

        recs = []
        popularity_rank = 0
        while len(recs) < N:
            if self.songs_ranked[popularity_rank] not in songs_listened_to:
                recs.append(self.songs_ranked[popularity_rank])
            popularity_rank += 1

        return recs

    def fit(self, train_csr):
        #csr matrix of item by users in same format as fit for CF model
        item_user = train_csr.tocoo()
        all_songs = item_user.row
        #remove duplicate song_sparse_id_entries
        self.unique_songs = set(all_songs)
        self.user_to_listened_songs_map = build_songs_listened_to_map(item_user)
        #get ranked popularity
        song_counts = Counter(all_songs).most_common()
        self.songs_ranked = [song_tuple[0] for song_tuple in song_counts]
