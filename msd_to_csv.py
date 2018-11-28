# Adapted from https://github.com/eltonlaw/msdvis/blob/d0f3c4511dc84e07f306ab219348ccd6fc3933df/grab_data.py

import os
import numpy as np
import sys
import pandas as pd
import sqlite3
import multiprocessing
import hdf5_getters


def get_song_data(categories,
                  path="./MillionSongSubset/data",
                  write=False,
                  write_path="song_data.csv",
                  start_i=0,
                  end_i=10000):
  """ Pulls data from all h5 files on the provided path and all it's child nodes
    Parameters
    ---------
    categories: List/Strings
        Strings are specific keywords. For an example of available categories:
        http://labrosa.ee.columbia.edu/millionsong/pages/example-track-description
    path: String,optional
        Path to the h5 data to decode and compile
    write: Boolean,optional
        If true, writes data to .csv at 'write_path'
    write_path: String,optional
        Destination of the final compiled pandas Dataframe.
    Returns
    ------
    Pandas Dataframe containing categories
    """
  # Conditional differentiates between different directory structures,
  # anything less than 30 means the path is referencing a parent directory
  h5_files = [x for x in os.walk(path) if len(x[0]) == 30]
  data, file_paths = [], []
  count = 0
  for root, dirs, files in h5_files:
    for f in files:
      file_paths.append(os.path.join(root, f))

  for file_path in file_paths[start_i:end_i]:
    h5file = hdf5_getters.open_h5_file_read(file_path)
    datapoint = {}
    for cat in categories:
      datapoint[cat] = getattr(hdf5_getters, "get_" + cat)(h5file)
    h5file.close()
    data.append(datapoint)

    progress = (count * 100.) / 10000  # 10 000 datapoints in sample
    sys.stdout.write(
        "==== Compiling subset of data...[ {:.2f}% ] ==== \r".format(progress))
    sys.stdout.flush()
    count += 1
  df = pd.DataFrame(data)
  if write:
    df.to_csv(write_path)
    print("Data written to {0}".format(write_path))
  return df


if __name__ == "__main__":
  # taken from https://labrosa.ee.columbia.edu/millionsong/pages/example-track-description
  categories_to_get = [
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
  print("...running grab_data.py")
  CPU_COUNT = multiprocessing.cpu_count()
  p = multiprocessing.Pool(processes=CPU_COUNT)
  p.apply_async(get_song_data(categories_to_get, write=True))