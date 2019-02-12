import pandas as pd
import numpy as np

songs_df = pd.read_hdf('data/full_msd_with_audio_features.h5', key='df')

# NOTE: if song_ids are passed in as normal strings, replace instances of song_id with prepare_song_id(song_id)

def prepare_song_id(song_id):
  return ('b\'' + song_id + '\'')


def get_diversity(song_vector_list):
  dissim = 0
  n = len(song_vector_list)
  print(n)
  print(n / 2)
  print(((n / 2) * (n - 1)))
  for i in range(n):
    for j in range(n):
      dissim += np.linalg.norm(song_vector_list[i] - song_vector_list[j])

  return dissim / ((n / 2) * (n - 1))


def convert_recs_to_embeddings(song_ids):
  embeddings_list = []

  for song_id in song_ids:
    # NOTE: if song_ids are passed in as normal strings, replace all following instances of song_id with prepare_song_id(song_id)
    if (song_id in songs_df.song_id.values):
      row = songs_df.loc[songs_df['song_id'] == song_id]
      embedding = row.iloc[:, 12:]
      embedding = np.array(embedding.values.tolist()[0])
      embeddings_list.append(embedding)

  return embeddings_list


def get_diversity_score(song_ids):
  # convert to list of embeddings
  embedding_list = convert_recs_to_embeddings(song_ids)

  # calculate diversity
  return get_diversity(embedding_list)


song_ids = list(songs_df.head(100).song_id)
print(len(song_ids))
print(get_diversity_score(song_ids))

feature_MUSIC_dict = {
    'danceability': np.array([-0.37, 0.05, -0.35, 0.08, 0.43]),
    'energy': np.array([-0.64, -0.46, -0.13, 0.66, -0.03]),
    'instrumentalness': np.array([0.20, -0.47, 0.28, 0.09, -0.01]),
    'liveness': np.array([-0.69, -0.12, -0.07, 0.43, 0.02]),
    'loudness': np.array([-0.58, -0.19, -0.44, 0.79, -0.21]),
    'valence': np.array([-0.04, 0.18, 0.24, -0.34, 0.18]),
}
