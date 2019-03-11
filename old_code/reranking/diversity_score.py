import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import pdist

embedding_cols = [
#     'year',
    'acousticness',
    'danceability',
    'duration_ms',
    'energy',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mode',
    'speechiness',
    'tempo',
    'time_signature',
    'valence'
]
msd = pd.read_hdf('../data/full_msd_with_audio_features.h5', key='df')
msd = msd[['song_id'] + embedding_cols]
msd.head()

def get_list_dissimilarity(song_ids):
    song_vectors = msd.loc[msd['song_id'].isin(song_ids)][embedding_cols].values
    return np.mean(pdist(song_vectors, 'cosine'))

song_ids = list(msd.head(1000).song_id)
start = time.time()
print(f'Dissim was {get_list_dissimilarity(song_ids)}')
print(f'It took {time.time() - start}s')