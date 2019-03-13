import numpy as np
from collections import Counter
a = [1, 2, 2, 2, 3, 3]
# list of tuples of format [(count, element)]
a_counts = Counter(a).most_common(2)
print(a_counts)
print(np.array(a_counts))
# top_m_songs = [i[0] for i in ]

# MUSIC = [2.651562142519999,
#  0.8866577970780001,
#  2.5151982915280002,
#  -4.368227499866,
#  1.6471934888740005]