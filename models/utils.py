import numpy as np


def build_songs_listened_to_map(item_user_coo):
    user_to_listened_songs_map = {}
    for song_index, user_index in zip(item_user_coo.row, item_user_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)

    return user_to_listened_songs_map


def weight_cf_matrix(csr_mat, alpha):
    #don't want to modify original incase it gets put into other models
    weighted_csr_mat = csr_mat.copy()
    weighted_csr_mat.data = 1 + np.log(alpha * csr_mat.data)
    return weighted_csr_mat