import numpy as np
import pandas as pd

feature_MUSIC_dict = {
    'danceability': np.array([-0.37, 0.05, -0.35, 0.08, 0.43]),
    'energy': np.array([-0.64, -0.46, -0.13, 0.66, -0.03]),
    'instrumentalness': np.array([0.20, -0.47, 0.28, 0.09, -0.01]),
    'liveness': np.array([-0.69, -0.12, -0.07, 0.43, 0.02]),
    'loudness': np.array([-0.58, -0.19, -0.44, 0.79, -0.21]),
    'valence': np.array([-0.04, 0.18, 0.24, -0.34, 0.18]),
}
feature_MUSIC_matrix = [MUSIC for MUSIC in feature_MUSIC_dict.values()]

msd = pd.read_hdf('data/full_msd_with_audio_features.h5', key='df')
msd = msd[['song_id'] + list(feature_MUSIC_dict.keys())]


def get_MUSIC(song_ids):
    song_vectors = msd.loc[msd['song_id'].isin(song_ids)][list(
        feature_MUSIC_dict.keys())].values
    return np.sum(
        np.dot(song_vectors, feature_MUSIC_matrix), axis=0) / len(song_ids)