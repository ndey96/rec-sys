from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from multiprocessing import Pool
import time
import pandas as pd
from scipy.sparse import load_npz
from ALSpkNN import get_baseline_cf_model, weight_cf_matrix
from functools import partial
import os
# https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx

# http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html


def py_mean_average_precision_at_k(model,
                                   train_user_items,
                                   test_user_items,
                                   K=10):

    user_items_coo = test_user_items.tocoo()
    user_to_listened_songs_map = {}
    for user_index, song_index in zip(user_items_coo.row, user_items_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)

    average_precision_sum = 0
    for user_index in user_to_listened_songs_map.keys():
        listened_song_indices = user_to_listened_songs_map[user_index]
        recommended_song_indices = user_to_recs_map[user_index]

        start = time.time()
        precision_sum = 0
        for k in range(1, K):
            num_correct_recs = len(listened_song_indices.intersection(recommended_song_indices[:k]))
            precision_sum += num_correct_recs / k

        average_precision_sum += precision_sum / min(K, len(listened_song_indices))
    
    return average_precision_sum / len(user_to_listened_songs_map)




def get_user_recs(user_index, K, model, train_user_items):
    recommended_song_indices = [
        rec[0]
        for rec in model.recommend(user_index, train_user_items, N=K)
    ]
    return recommended_song_indices

K = None
model = None
train_user_items = None

def initializer(K_init, model_init, train_user_items_init):
    global K
    K = K_init
    global model
    model = model_init
    global train_user_items
    train_user_items = train_user_items_init

def get_user_recs_wrapper(user_index):
    return get_user_recs(user_index, K, model, train_user_items)

def multi_mean_average_precision_at_k(model,
                                      train_user_items,
                                      test_user_items,
                                      K=10):
    user_items_coo = test_user_items.tocoo()

    # user_to_listened_songs_map -> {user_index: listened_song_indices}
    user_to_listened_songs_map = {}
    for user_index, song_index in zip(user_items_coo.row, user_items_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)

    rec_pool = Pool(os.cpu_count(), initializer, (K, model, train_user_items))
    start = time.time()
    print('Starting pool.map')
    # user_recs -> [recommended_song_indices] -> index of element corresponds to user_index position
    user_recs = rec_pool.map(
        func=get_user_recs_wrapper,
        iterable=list(user_to_listened_songs_map.keys())[:10000],
        # iterable=user_to_listened_songs_map.keys(),
        chunksize=625
    )
    print(f'recs: {time.time() - start}s')

    start = time.time()
    average_precision_sum = 0
    for i, user_index in enumerate(user_to_listened_songs_map.keys()):
        listened_song_indices = user_to_listened_songs_map[user_index]
        recommended_song_indices = user_recs[i]

        start = time.time()
        precision_sum = 0
        for k in range(1, K):
            num_correct_recs = len(listened_song_indices.intersection(recommended_song_indices[:k]))
            precision_sum += num_correct_recs / k

        average_precision_sum += precision_sum / min(K, len(listened_song_indices))
    
    print(f'MAP: {time.time() - start}s')
    return average_precision_sum / len(user_to_listened_songs_map)

if __name__ == '__main__':
    # user_df = pd.read_hdf('data/user_df.h5', key='df')[['user_id', 'sparse_index', 'MUSIC', 'song_ids']]
    # # user_df = pd.read_hdf('data/user_df.h5', key='df')
    # user_df.set_index('sparse_index', inplace=True)

    # song_df = pd.read_hdf('data/song_df.h5', key='df')[['song_id', 'sparse_index']]
    # # song_df = pd.read_hdf('data/song_df.h5', key='df')
    # song_df.set_index('song_id', inplace=True)
    # train_plays = load_npz('data/train_sparse.npz')
    # test_plays = load_npz('data/test_sparse.npz')

    # from ALSpkNN import ALSpkNN

    # print("Building model...")
    # model = ALSpkNN(user_df, song_df, k=100, knn_frac=0.5, cf_weighting_alpha=1)
    # print("Fitting model...")
    # model.fit(train_plays)
    # recs = model.recommend(user_sparse_index=12345, train_plays_transpose=train_plays.transpose(), N=5)
    # print(recs)

    # start = time.time()
    # MAPk = multi_mean_average_precision_at_k(
    #     model,
    #     train_user_items=train_plays.transpose(),
    #     test_user_items=test_plays.transpose(),
    #     K=5)
    # #     show_progress=False,
    # #     num_threads=0)

    # print("MAPK for ALSpKNN is: " + str(MAPk))
    # print(f'Calculation took {time.time() - start}s')

    ##################################################################

    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')

    # print("Building and fitting the baseline CF model")
    baseline_cf_model = get_baseline_cf_model()
    # weighted_train_csr = weight_cf_matrix(train_plays, alpha=1)
    baseline_cf_model.fit(train_plays)

    # print("Evaluating the baseline CF model")
    start = time.time()

    MAPk = multi_mean_average_precision_at_k(
        model=baseline_cf_model,
        train_user_items=train_plays.transpose(),
        test_user_items=test_plays.transpose(),
        K=5)

    print("MAPK for baseline CF is: " + str(MAPk))
    print(f'Calculation took {time.time() - start}s')

