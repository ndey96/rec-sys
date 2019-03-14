import numpy as np
import random

    #In Get Metrics code, user_item_coo is the transpose of what is being passed in here.
def build_songs_listened_to_map(item_user_coo):
    user_to_listened_songs_map = {}
    for song_index,user_index in zip(item_user_coo.row, item_user_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)
        
    return user_to_listened_songs_map

class RandomRecommender:
    def recommend(self, N,user_sparse_index,train_plays_transpose=None):
        if user_sparse_index in self.user_to_listened_songs_map:
            songs_listened_to = self.user_to_listened_songs_map[user_sparse_index]
        else:
            songs_listened_to = set()
        possible_songs = self.unique_songs - songs_listened_to
        recs = random.sample(possible_songs,N)
        return recs

    def fit(self, train_csr):
        #csr matrix of item by users in same format as fit for CF model
        item_user= train_csr.tocoo()
        all_songs = item_user.row
        #remove duplicate song_sparse_id_entries
        self.unique_songs = set(all_songs)
        self.user_to_listened_songs_map = build_songs_listened_to_map(item_user)