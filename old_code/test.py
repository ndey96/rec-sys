from multiprocessing import Pool
import time
from tqdm import tqdm
import pandas as pd


def f(x):
  # time.sleep(1)
  return {
      '1': 1 * x,
      '2': 2 * x,
      '3': 3 * x,
      '4': 4 * x,
      '5': 5 * x,
      '6': 6 * x,
      '7': 7 * x,
      '8': 8 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
      '9': 9 * x,
  }


if __name__ == '__main__':
  n = 1000000

  start = time.time()
  df = pd.DataFrame({})
  for datapoint in tqdm(Pool().imap(f, range(n)), total=n):
    df.append(datapoint, ignore_index=True)
  print(time.time() - start)

  start = time.time()
  df = pd.DataFrame(Pool().map(f, range(n), chunksize=625))
  print(time.time() - start)

  start = time.time()
  data = list(tqdm(Pool().imap_unordered(f, range(n), chunksize=625), total=n))
  print(data[0])
  df = pd.DataFrame(data)
  print(time.time() - start)

  # n = 1000000
  # start = time.time()
  # data = list(tqdm(Pool().imap(f, range(n), chunksize=1), total=n))
  # print(data[0])
  # print(time.time() - start)
