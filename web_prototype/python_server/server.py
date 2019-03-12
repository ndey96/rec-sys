from flask import Flask, jsonify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from ALSpkNN import ALSpkNN
from flask_cors import CORS


def get_MUSIC(sub_df):
    # LET n = number of songs
    # LET m = number of audio features
    feature_MUSIC_dict = {
        'danceability': np.array([-0.37, 0.05, -0.35, 0.08, 0.43]),
        'energy': np.array([-0.64, -0.46, -0.13, 0.66, -0.03]),
        'instrumentalness': np.array([0.20, -0.47, 0.28, 0.09, -0.01]),
        'liveness': np.array([-0.69, -0.12, -0.07, 0.43, 0.02]),
        'loudness': np.array([-0.58, -0.19, -0.44, 0.79, -0.21]),
        'valence': np.array([-0.04, 0.18, 0.24, -0.34, 0.18]),
    }
    # feature_MUSIC_matrix -> m x 5 matrix, where m is the number of audio features in feature_MUSIC_dict
    feature_MUSIC_matrix = [MUSIC for MUSIC in feature_MUSIC_dict.values()]
    # song_vectors -> n x m matrix, where m is the number of audio features in feature_MUSIC_dict
    song_vectors = sub_df[list(feature_MUSIC_dict.keys())].values

    # unweighted_MUSIC_vals -> n x 5 matrix
    unweighted_MUSIC_vals = song_vectors @ feature_MUSIC_matrix

    return list(np.mean(unweighted_MUSIC_vals, axis=0))


app = Flask(__name__)
CORS(app)


@app.route("/getrecd/<sp_ids_string>")
def GET(sp_ids_string):
    sp_ids = sp_ids_string.split(',')
    client_credentials_manager = SpotifyClientCredentials(
        client_id='02c386b9bcfb43248ad7011c576c1e3f',
        client_secret='f2406ab918d349689325d2f6916aa656')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    audio_features = sp.audio_features(tracks=sp_ids)

    audio_df = pd.DataFrame(audio_features)
    MUSIC = get_MUSIC(audio_df)
    user_df = pd.read_hdf('user_df.h5')

    user_sparse_index = 69696969
    user_row = pd.DataFrame([{
        'sparse_index': user_sparse_index,
        'user_id': 'spotify_user',
        'MUSIC': MUSIC,
        'num_songs': 0,
        'is_test': False,
        'song_sparse_indices': []
    }])
    user_row.set_index('sparse_index', inplace=True)
    # user_row.head()
    user_df = user_df.append(user_row)
    song_df = pd.read_hdf('song_df.h5', key='df')

    model = ALSpkNN(user_df, song_df, k=35, knn_frac=1)
    song_sparse_indices = model.recommend(
        user_sparse_index=user_sparse_index, train_plays_transpose=None, N=20)

    spotify_song_ids = song_df.loc[song_sparse_indices]['spotify_id'].to_list()

    return jsonify(result=spotify_song_ids)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)