from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from multiprocessing import Pool
import time
import pandas as pd
from scipy.sparse import load_npz
from ALSpkNN import get_baseline_cf_model, weight_cf_matrix, ALSpkNN
from functools import partial
import os
from sklearn import preprocessing

# https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx

# http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

def get_user_recs(user_index, K, model, train_user_items):
    return model.recommend(user_index, train_user_items, N=K)

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

def mean_average_precision_at_k(user_recs,
                                user_to_listened_songs_map,
                                K,
                                limit):

    average_precision_sum = 0
    for i, user_index in enumerate(list(user_to_listened_songs_map.keys())[:limit]):
        listened_song_indices = user_to_listened_songs_map[user_index]
        recommended_song_indices = user_recs[i]

        precision_sum = 0
        for k in range(1, K):
            num_correct_recs = len(listened_song_indices.intersection(recommended_song_indices[:k]))
            precision_sum += num_correct_recs / k

        average_precision_sum += precision_sum / min(K, len(listened_song_indices))
    
    return average_precision_sum / len(user_to_listened_songs_map)

# TODO: use play counts and scale song_vectors before calculating pdist
# TOOD: refactor this code to work in this file!
def get_cosine_list_dissimilarity(user_recs,
                                  K,
                                  limit,
                                  song_df):
    embedding_cols = [
        # 'year',
        'acousticness',
        'danceability',
        'duration_ms',
        'energy',
        'instrumentalness',
        'key',
        'liveness',
        'loudness',
        'mode',
        'speechiness',
        'tempo',
        'time_signature',
        'valence'
    ]

    song_df[embedding_cols] = preprocessing.MinMaxScaler().fit_transform(song_df[embedding_cols])

    list_dissim_sum = 0
    for recommended_song_indices in user_recs:
        # song_vectors -> n x m matrix, where m is the number of audio features in the embedding_cols
        song_vectors = song_df[recommended_song_indices][embedding_cols].values
        if len(song_vectors) == 1:
            continue
        list_dissim_sum += np.mean(pdist(song_vectors, 'cosine'))

    return list_dissim_sum/len(user_recs)

# eg: metrics = ['MAP@K', 'cosine_list_dissimilarity']
# N = number of recommendations per user
def get_metrics(
    metrics,
    N,
    model,
    train_user_items,
    test_user_items,
    song_df,
    limit):
    
    user_items_coo = test_user_items.tocoo()

    # user_to_listened_songs_map -> {user_index: listened_song_indices}
    user_to_listened_songs_map = {}
    for user_index, song_index in zip(user_items_coo.row, user_items_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)

    rec_pool = Pool(os.cpu_count(), initializer, (N, model, train_user_items))
    start = time.time()
    print('Starting pool.map')
    # user_recs -> [recommended_song_indices] -> index of element corresponds to user_index position
    user_recs = rec_pool.map(
        func=get_user_recs_wrapper,
        iterable=list(user_to_listened_songs_map.keys())[:limit],
        # iterable=user_to_listened_songs_map.keys(),
        chunksize=625
    )
    print(f'recs time: {time.time() - start}s')

    calculated_metrics = {}
    if 'MAP@K' in metrics:
        start = time.time()
        map_at_k = mean_average_precision_at_k(
            user_recs=user_recs,
            user_to_listened_songs_map=user_to_listened_songs_map,
            model=model,
            train_user_items=train_user_items,
            test_user_items=test_user_items,
            K=N,
            limit=limit)
        calculated_metrics['MAP@K'] = map_at_k
        print(f'MAP@K calculation time: {time.time() - start}s')

    if 'cosine_list_dissimilarity' in metrics:
        start = time.time()
        calculated_metrics['cosine_list_dissimilarity'] = 69
        print(f'MAP@K calculation time: {time.time() - start}s')

    return calculated_metrics

if __name__ == '__main__':

    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')

    print("Building and fitting the baseline CF model")
    baseline_cf_model = get_baseline_cf_model()
    # weighted_train_csr = weight_cf_matrix(train_plays, alpha=1)
    baseline_cf_model.fit(train_plays)

    print("Evaluating the baseline CF model")
    metrics = get_metrics(
        metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
        N=5,
        model=baseline_cf_model,
        train_user_items=train_plays.transpose(),
        test_user_items=test_plays.transpose(),
        song_df_sparse_indexed=None,
        limit=1000)
    print(metrics)

    ##################################################################

    # user_df = pd.read_hdf('data/user_df.h5', key='df')[['user_id', 'sparse_index', 'MUSIC', 'song_ids']]
    # # user_df = pd.read_hdf('data/user_df.h5', key='df')
    # user_df.set_index('sparse_index', inplace=True)

    # song_df = pd.read_hdf('data/song_df.h5', key='df')[['song_id', 'sparse_index']]
    # # song_df = pd.read_hdf('data/song_df.h5', key='df')
    # song_df.set_index('song_id', inplace=True)
    # train_plays = load_npz('data/train_sparse.npz')
    # test_plays = load_npz('data/test_sparse.npz')

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

    # print("MAPK for ALSpKNN is: " + str(MAPk))
    # print(f'Calculation took {time.time() - start}s')

