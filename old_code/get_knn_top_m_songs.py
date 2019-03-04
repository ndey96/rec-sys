import pandas as pd
from scipy.spatial import KDTree
from random import randint
import csv
from collections import Counter
import sys


def load_user_MUSIC_df():

    user_MUSIC_df = pd.read_hdf('data/user_div_MUSIC_divpref.h5', key='df')

    return user_MUSIC_df


def load_user_playlist_df():

    user_playlist_df = pd.read_hdf('data/userid_playlist.h5', key='df')

    return user_playlist_df


def build_tree(user_MUSIC_df):

    MUSIC_vectors = user_MUSIC_df.iloc[:, 0].values.tolist()
    tree = KDTree(MUSIC_vectors)

    return tree


def get_closest_MUSIC_userids(userid, k, user_MUSIC_df):

    user_MUSIC = user_MUSIC_df.loc[user_MUSIC_df['user_id'] == userid][
        'MUSIC'].tolist()[0]

    distances, indices = tree.query(user_MUSIC, k)
    closest_userids = []

    for index in indices:
        closest_userids.append(user_MUSIC_df.iloc[index, 3])

    return closest_userids


def get_knn_top_m_songs(userid, k, m, tree, user_playlist_df, user_MUSIC_df):

    closest_userids = get_closest_MUSIC_userids(userid, k, user_MUSIC_df)

    closest_user_songs = []
    for i in range(len(closest_userids)):
        closest_user_songs.append(user_playlist_df.loc[user_playlist_df[
            'user_id'] == closest_userids[i]]['playlist'].tolist()[0])

    closest_user_songs = [
        item for sublist in closest_user_songs for item in sublist
    ]
    counted_closest_user_songs = Counter(closest_user_songs)
    top_m_songs = [i[0] for i in counted_closest_user_songs.most_common()[:m]]

    return top_m_songs


if __name__ == "__main__":

    user_playlist_df = load_user_playlist_df()
    user_MUSIC_df = load_user_MUSIC_df()
    tree = build_tree()

    get_knn_top_m_songs(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), tree,
                        user_playlist_df, user_MUSIC_df)