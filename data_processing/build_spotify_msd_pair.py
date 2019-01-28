# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:15:12 2019

@author: Matthew and Nolan
"""

import json
import os
import pandas as pd
from multiprocessing import Pool

def get_id_pair(file_path):
  with open(file_path) as f:
    data_all = json.load(f)['response']['songs']
    if (len(data_all) == 0):
      # this is the wierd and rare case that a song on MSD has not matches on spotify
      return {'spotify_id': None, 'msd_id': file_path[-23:-5]}

    data = data_all[0]
    MSDsongid = data['id']
    for trackPackage in data['tracks']:
      if trackPackage['catalog'] == 'spotify':
        spotify_id = trackPackage['foreign_id'].split(":")[2]
        return {'spotify_id': spotify_id, 'msd_id': MSDsongid}

    return {'spotify_id': 'None', 'msd_id': MSDsongid}

if __name__ == "__main__":
  base_path = 'millionsongdataset_echonest/'
  write_path = 'spotify_msd_id_pairs.csv'
  json_files = [x for x in os.walk(base_path)]
  file_paths = []
  for root, dirs, files in json_files:
    for f in files:
      file_paths.append(os.path.join(root, f))

  print(file_paths[0])
  print(get_id_pair(file_paths[0]))
  print(len(file_paths))
  print('Extracting data from json files now...')
  data = Pool().map(func=get_id_pair, iterable=file_paths, chunksize=625)
  print('Data extracted into list. Creating dataframe now...')
  df = pd.DataFrame(data)
  print(df.shape)
  df = df.dropna()
  print(df.shape)
  print('Dataframe created. Writing to csv now...')
  df.to_csv(write_path, index=None)
  print(f'Data written to {write_path}')