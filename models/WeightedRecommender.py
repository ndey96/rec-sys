# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:56:23 2019

@author: Matthew
"""
import numpy as np
import random
from collections import Counter
from .utils import build_songs_listened_to_map


class WeightedRecommender:

    def recommend(self, user_sparse_index, train_plays_transpose=None, N=20):
        if user_sparse_index in self.user_to_listened_songs_map:
            songs_listened_to = self.user_to_listened_songs_map[
                user_sparse_index]
        else:
            songs_listened_to = []

        matching_indices = []
        for ind, song_id in enumerate(self.songs_ranked):
            if song_id in songs_listened_to:
                matching_indices.append(ind)

        # make a copy of array so you don't delete songs everytime recommend is ran
        possible_songs = self.songs_ranked[:]
        play_counts = self.songs_plays_ranked[:]
        for ind in matching_indices:
            possible_songs.pop(ind)
            play_counts.pop(ind)

        song_probs = np.array(play_counts)
        song_probs = song_probs / song_probs.sum()

        recs = np.random.choice(
            a=possible_songs, p=song_probs, size=N, replace=False)
        return recs

    def fit(self, train_csr):
        #csr matrix of item by users in same format as fit for CF model
        item_user = train_csr.tocoo()
        all_songs = item_user.row
        #remove duplicate song_sparse_id_entries
        #        self.unique_songs = set(all_songs)
        self.user_to_listened_songs_map = build_songs_listened_to_map(item_user)

        song_counts = Counter(all_songs).most_common()
        self.songs_ranked = [song_tuple[0] for song_tuple in song_counts]
        self.songs_plays_ranked = [song_tuple[1] for song_tuple in song_counts]