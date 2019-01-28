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
  # train_triplets.txt downloaded from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip
  with open('train_triplets.txt') as f:
    lines = f.readlines()

  print('Lines extracted into list. Creating list of dicts now...')
  data = Pool().map(func=get_dict_from_line, iterable=lines, chunksize=625)
  print('List of dicts created. Creating dataframe now...')
  df = pd.DataFrame(data).dropna()
  print('Dataframe created. Writing to csv now...')
  df.to_hdf('triplets.h5', key='df', mode='w')