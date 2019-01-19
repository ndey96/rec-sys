from multiprocessing import Pool
import time
from tqdm import tqdm


def f(x):
  # time.sleep(1)
  return x * x


if __name__ == '__main__':
  start = time.time()
  print(list(tqdm(Pool().imap(f, range(100)), total=100)))

  # print(p.map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
  print(time.time() - start)
