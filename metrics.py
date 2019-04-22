from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist
from multiprocessing import Pool
import time
import pandas as pd
from scipy.sparse import load_npz
from functools import partial
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt

def get_user_recs(user_index, K, model, train_user_items):
    return model.recommend(user_index, train_user_items, N=K)

K = None
model = None
train_user_items = None

def recs_initializer(K_init, model_init, train_user_items_init):
    global K
    K = K_init
    global model
    model = model_init
    global train_user_items
    train_user_items = train_user_items_init

def get_user_recs_wrapper(user_index):
    return get_user_recs(user_index, K, model, train_user_items)

def get_user_list_dissim(recommended_song_indices, song_df, embedding_cols):
    song_vectors = song_df.loc[recommended_song_indices][embedding_cols].values
    if len(song_vectors) == 1:
        return 0
    return np.mean(pdist(song_vectors, 'cosine'))

song_df = None
embedding_cols = None

def list_dissim_initializer(song_df_init, embedding_cols_init):
    global song_df
    song_df = song_df_init
    global embedding_cols
    embedding_cols = embedding_cols_init

def get_user_list_dissim_wrapper(recommended_song_indices):
    return get_user_list_dissim(recommended_song_indices, song_df, embedding_cols)

def get_mean_cosine_list_dissimilarity(user_recs,
                                       K,
                                       limit,
                                       song_df):
    embedding_cols = [
        # 'year',
        'acousticness',
        'danceability',
        # 'duration_ms',
        'energy',
        'instrumentalness',
        # 'key',
        'liveness',
        'loudness',
        # 'mode',
        'speechiness',
        'tempo',
        # 'time_signature',
        'valence'
    ]

    song_df[embedding_cols] = preprocessing.MinMaxScaler().fit_transform(song_df[embedding_cols])
    with Pool(os.cpu_count(), list_dissim_initializer, (song_df, embedding_cols)) as dissim_pool:
        list_dissims = dissim_pool.map(
            func=get_user_list_dissim_wrapper,
            iterable=user_recs[:limit],
            chunksize=625
        )
    return np.mean(list_dissims)

song_df = None

def meta_div_initializer(song_df_init):
    global song_df
    song_df = song_df_init

def get_user_meta_div_wrapper(recommended_song_indices):
    return get_user_meta_div(recommended_song_indices, song_df)

def get_user_meta_div(recommended_song_indices, song_df):
    # calculated using 10k users
    num_genre_avg = 2.4 
    num_artist_avg = 16.2
    year_std_avg = 5.5

    sub_df = song_df.loc[recommended_song_indices]
    genre_diversity = sub_df['genre'].nunique() / num_genre_avg
    artist_diversity = sub_df['artist_name'].nunique() / num_artist_avg
    era_diversity = (sub_df['year'].where(sub_df['year'] > 0)).std() / year_std_avg
    
    return genre_diversity + artist_diversity + era_diversity

def get_mean_metadata_diversity(user_recs, song_df, limit):
    
    scaling_factor = 20 / len(user_recs[0])
    
    with Pool(os.cpu_count(), meta_div_initializer, (song_df,)) as meta_div_pool:
        meta_divs = meta_div_pool.map(
            func=get_user_meta_div_wrapper,
            iterable=user_recs[:limit],
            chunksize=625
        )
    
    return scaling_factor * np.mean(meta_divs)

def get_user_num_genres_wrapper(recommended_song_indices):
    return get_user_num_genres(recommended_song_indices, song_df)

def get_user_num_genres(recommended_song_indices, song_df):
    sub_df = song_df.loc[recommended_song_indices]
    return sub_df['genre'].nunique()

def get_mean_num_genres(user_recs, song_df, limit):    
    with Pool(os.cpu_count(), meta_div_initializer, (song_df,)) as num_genres_pool:
        num_genres = num_genres_pool.map(
            func=get_user_num_genres_wrapper,
            iterable=user_recs[:limit],
            chunksize=625
        )
    
    return np.mean(num_genres)


def get_mean_average_precision_at_k(user_recs,
                                    user_to_listened_songs_map,
                                    K,
                                    limit):
    # http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
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

# eg: metrics = ['MAP@K', 'mean_cosine_list_dissimilarity']
# N = number of recommendations per user
def get_metrics(
    metrics,
    N,
    model,
    train_user_items,
    test_user_items,
    song_df,
    limit=9999999):
    
    user_items_coo = test_user_items.tocoo()

    # user_to_listened_songs_map -> {user_index: listened_song_indices}
    user_to_listened_songs_map = {}
    for user_index, song_index in zip(user_items_coo.row, user_items_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)

    start = time.time()
    print('Starting pool.map')
    with Pool(os.cpu_count(), recs_initializer, (N, model, train_user_items)) as rec_pool:
        # user_recs -> [recommended_song_indices] -> index of element corresponds to user_index position
        user_recs = rec_pool.map(
            func=get_user_recs_wrapper,
            iterable=list(user_to_listened_songs_map.keys())[:limit],
            # iterable=user_to_listened_songs_map.keys(),
            chunksize=625
        )
    if isinstance(user_recs[0][0], tuple):
        new_user_recs = []
        for user in user_recs:
            recs_for_user = []
            for rec in user:
                recs_for_user.append(rec[0])
            new_user_recs.append(recs_for_user)
        user_recs = new_user_recs

    print(f'recs time: {time.time() - start}s')

    calculated_metrics = {}
    if 'MAP@K' in metrics:
        start = time.time()
        map_at_k = get_mean_average_precision_at_k(
            user_recs=user_recs,
            user_to_listened_songs_map=user_to_listened_songs_map,
            K=N,
            limit=limit)
        calculated_metrics['MAP@K'] = map_at_k
        print(f'MAP@K calculation time: {time.time() - start}s')

    if 'mean_cosine_list_dissimilarity' in metrics:
        start = time.time()
        cos_dis = get_mean_cosine_list_dissimilarity(user_recs=user_recs,
                                                K=K,
                                                limit=limit,
                                                song_df=song_df)
        calculated_metrics['mean_cosine_list_dissimilarity'] = cos_dis
        print(f'mean_cosine_list_dissimilarity calculation time: {time.time() - start}s')

    if 'metadata_diversity' in metrics:
        start = time.time()
        metadata_diversity = get_mean_metadata_diversity(user_recs=user_recs,
                                                    limit=limit,
                                                    song_df=song_df)
        calculated_metrics['metadata_diversity'] = metadata_diversity
        print(f'metadata_diversity calculation time: {time.time() - start}s')

    if 'num_genres' in metrics:
        start = time.time()
        num_genres = get_mean_num_genres(user_recs=user_recs,
                                        limit=limit,
                                        song_df=song_df)
        calculated_metrics['num_genres'] = num_genres
        print(f'num_genres calculation time: {time.time() - start}s')

    return calculated_metrics

if __name__ == '__main__':
    from models import ALSpkNN, ALSRecommender, PopularRecommender, RandomRecommender, WeightedRecommender
    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')
    song_df = pd.read_hdf('data/song_df.h5', key='df')
    user_df = pd.read_hdf('data/user_df.h5', key='df')
    metrics_to_calc = ['MAP@K','num_genres']
    hparam_vals = {
        'k': 30,
        'max_overlap': 0.2,
        'knn_frac': 0.5,
        'min_songs': 5,
        'cf_weighting_alpha': 1,
        'mode': 'weighted_random',
        'bottom_branch': 'ALS',
    }
    print(f"Building and fitting the ALSpkNN model with {hparam_vals}")
    model = ALSpkNN(user_df, song_df, **hparam_vals)
    model.fit(train_plays)
    print("Evaluating the ALSpkNN model")
    metrics = get_metrics(
        metrics=metrics_to_calc,
        N=20,
        model=model,
        train_user_items=train_plays.transpose(),
        test_user_items=test_plays.transpose(),
        song_df=song_df,
        limit=99999999)
    print(metrics)
    # song_sparse_indices = model.recommend(
    #     user_sparse_index=1234, train_plays_transpose=train_plays.transpose(), N=20)
    # print(song_sparse_indices)
    # assert len(song_sparse_indices) == len(np.unique(song_sparse_indices))