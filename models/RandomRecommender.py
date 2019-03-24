import numpy as np
import random
from .utils import build_songs_listened_to_map


class RandomRecommender:

    def recommend(self, user_sparse_index, train_plays_transpose=None, N=20):
        if user_sparse_index in self.user_to_listened_songs_map:
            songs_listened_to = self.user_to_listened_songs_map[
                user_sparse_index]
        else:
            songs_listened_to = set()
        possible_songs = self.unique_songs - songs_listened_to
        recs = random.sample(possible_songs, N)
        return recs

    def fit(self, train_csr):
        #csr matrix of item by users in same format as fit for CF model
        item_user = train_csr.tocoo()
        all_songs = item_user.row
        #remove duplicate song_sparse_id_entries
        self.unique_songs = set(all_songs)
        self.user_to_listened_songs_map = build_songs_listened_to_map(item_user)