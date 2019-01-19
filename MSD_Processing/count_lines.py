# Ex: python count_lines.py song_data.csv

import sys

if __name__ == '__main__':
  print(sys.argv[1])
  with open(sys.argv[1]) as f:
    print(sum(1 for line in f))