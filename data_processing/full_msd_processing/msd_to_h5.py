# Adapted from https://github.com/eltonlaw/msdvis/blob/d0f3c4511dc84e07f306ab219348ccd6fc3933df/grab_data.py

import os
import sys
import pandas as pd
import sqlite3
from multiprocessing import Pool
import hdf5_getters
from tqdm import tqdm

""" Pulls data from all h5 files on the provided path and all it's child nodes
    Parameters
    ---------
    categories: List/Strings
        Strings are specific keywords. For an example of available categories:
        http://labrosa.ee.columbia.edu/millionsong/pages/example-track-description
    path: String,optional
        Path to the h5 data to decode and compile
    write_path: String,optional
        Destination of the final compiled pandas Dataframe.
    Returns
    ------
    Pandas Dataframe containing categories
    """
  # Conditional differentiates between different directory structures,
  # anything less than 30 means the path is referencing a parent directory

# taken from https://labrosa.ee.columbia.edu/millionsong/pages/example-track-description
categories = [
      'artist_mbid',
      'artist_name',
      'artist_playmeid',
      'danceability',
      'duration',
      'energy',
      'key',
      'loudness',
      'mode',
      'release_7digitalid',
      'release',
      'song_hotttnesss',
      'song_id',
      'tempo',
      'time_signature',
      'title',
      'track_7digitalid',
      'track_id',
      'year',
]
path = "./MillionSongSubset/data"
write_path = "song_data.h5"
if len(sys.argv) > 1:
  path = sys.argv[1]
  write_path = sys.argv[2]

def get_datapoint(file_path):
      h5file = hdf5_getters.open_h5_file_read(file_path)
      datapoint = {}
      for cat in categories:
        datapoint[cat] = getattr(hdf5_getters, "get_" + cat)(h5file)
      h5file.close()
      return datapoint

if __name__ == "__main__":
  h5_files = [x for x in os.walk(path)]
  file_paths = []
  for root, dirs, files in h5_files:
    for f in files:
      file_paths.append(os.path.join(root, f))

  print(file_paths[0])
  print(len(file_paths))
  print('Extracting data from h5 files now...')
  data = Pool().map(func=get_datapoint, iterable=file_paths, chunksize=625)
  print('Data extracted into list. Creating dataframe now...')
  df = pd.DataFrame(data)
  print('Dataframe created. Writing to h5 file now...')
  df.to_hdf(write_path, key='df', mode='w')
  print(f'Data written to {write_path}')
