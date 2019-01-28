import pandas as pd
from multiprocessing import Pool


def get_dict_from_line(line):
  values = line[:-1].split('\t')
  if len(values) == 1:
    return {
        'user_id': None,
        'song_id': None,
        'play_count': None,
    }
  user_id, song_id, play_count = values
  return {
      'user_id': user_id,
      'song_id': song_id,
      'play_count': play_count,
  }


if __name__ == '__main__':
  # mismatches.txt downloaded from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/tasteprofile/sid_mismatches.txt
  with open('mismatches.txt') as f:
    lines = f.readlines()

  print('Lines extracted into list. Creating list of dicts now...')
  data = Pool().map(func=get_dict_from_line, iterable=lines, chunksize=625)
  print('List of dicts created. Creating dataframe now...')
  df = pd.DataFrame(data).dropna()
  print('Dataframe created. Writing to csv now...')
  df.to_hdf('msd_mismatches.h5', key='df', mode='w')
