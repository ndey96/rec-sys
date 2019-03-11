from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import pdist
from multiprocessing import Pool
import time
import pandas as pd
from scipy.sparse import load_npz
from ALSpkNN import get_baseline_cf_model, weight_cf_matrix, ALSpkNN
from functools import partial
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt

# https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx

# http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

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

def get_cosine_list_dissimilarity(user_recs,
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
    # for recommended_song_indices in user_recs[:limit]:
    #     # song_vectors -> n x m matrix, where m is the number of audio features in the embedding_cols
    #     song_vectors = song_df.loc[recommended_song_indices][embedding_cols].values
    #     if len(song_vectors) == 1:
    #         continue
    #     list_dissim_sum += np.mean(pdist(song_vectors, 'cosine'))

    # return list_dissim_sum/len(user_recs)



# eg: metrics = ['MAP@K', 'mean_cosine_list_dissimilarity']
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
        map_at_k = mean_average_precision_at_k(
            user_recs=user_recs,
            user_to_listened_songs_map=user_to_listened_songs_map,
            K=N,
            limit=limit)
        calculated_metrics['MAP@K'] = map_at_k
        print(f'MAP@K calculation time: {time.time() - start}s')

    if 'mean_cosine_list_dissimilarity' in metrics:
        start = time.time()
        cos_dis = get_cosine_list_dissimilarity(user_recs=user_recs,
                                                K=K,
                                                limit=limit,
                                                song_df=song_df)
        calculated_metrics['mean_cosine_list_dissimilarity'] = cos_dis
        print(f'mean_cosine_list_dissimilarity calculation time: {time.time() - start}s')

    return calculated_metrics

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

def get_custom_diversity_metric(user_recs):
    
#     num_genre_avg = 
#     num_artist_avg = 
#     year_std_avg = 
    
#     num_genres = 
#     num_artists = 
#     year_list = 
    
    genre_diversity = num_genres/num_genre_avg
    artist_diversity = num_artists/num_artist_avg
    era_diversity = np.std(year_list)/year_std_avg
    
    diversity = genre_diversity*0.5 + artist_diversity*0.25 + era_diversity*0.005
    
    return diversity

if __name__ == '__main__':

    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')
    song_df = pd.read_hdf('data/song_df.h5', key='df')
    user_df = pd.read_hdf('data/user_df.h5', key='df')

    ##################################################################

    # print("Building and fitting the baseline CF model")
    # baseline_cf_model = get_baseline_cf_model()
    # # weighted_train_csr = weight_cf_matrix(train_plays, alpha=1)
    # baseline_cf_model.fit(train_plays)

    # print("Evaluating the baseline CF model")
    # metrics = get_metrics(
    #     metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
    #     N=20,
    #     model=baseline_cf_model,
    #     train_user_items=train_plays.transpose(),
    #     test_user_items=test_plays.transpose(),
    #     song_df=song_df,
    #     limit=10000)
    # print(metrics)

    ##################################################################

    print("Building model...")
    NUM_VALS = 5
    model = ALSpkNN(user_df, song_df, k=100, knn_frac=0.25, cf_weighting_alpha=1, min_overlap=0.05)
    print("Fitting model...")
    model.fit(train_plays)
    
    all_MAPk_vals = []
    all_cos_dissim_vals = []
    limit_users = 5000
    
    MAPk_vals = np.zeros(NUM_VALS)
    cos_dissim_vals = np.zeros(NUM_VALS)
    
    knn_frac_vals = [0.1,0.3,0.5,0.7,0.9]
    for i,knn_frac_val in enumerate(knn_frac_vals):
        model.knn_frac = knn_frac_val
        metrics = get_metrics(
            metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
            N=20,
            model=model,
            train_user_items=train_plays.transpose(),
            test_user_items=test_plays.transpose(),
            song_df=song_df,
            limit=limit_users)
        MAPk_vals[i] = metrics["MAP@K"]
        cos_dissim_vals[i] = metrics["mean_cosine_list_dissimilarity"]
        print(metrics)
    
    #reset the model
    model.knn_frac = 0.25
    all_MAPk_vals.append(MAPk_vals)
    all_cos_dissim_vals.append(cos_dissim_vals)
    print(MAPk_vals)
    plt.figure()
    plt.title("MAP@K for varying MUSIC profile weight")
    plt.xlabel("Fraction of songs determined by MUSIC profile")
    plt.ylabel("MAP@K value")
    plt.plot(knn_frac_vals,MAPk_vals)
#     plt.show(block=False)
    plt.savefig("./figures/mapk_knn_frac")
    
    print(cos_dissim_vals)
    plt.figure()
    plt.title("Cosine list dissimilarity for varying MUSIC profile weight")
    plt.xlabel("Fraction of songs determined by MUSIC profile")
    plt.ylabel("List Dissimilarity")
    plt.plot(knn_frac_vals,cos_dissim_vals)
#     plt.show(block=False)
    plt.savefig("./figures/div_knn_frac")
    
    #redefine
    MAPk_vals = np.zeros(NUM_VALS)
    cos_dissim_vals = np.zeros(NUM_VALS)
    
    min_overlap_vals = [0, 0.05, 0.1,0.15,0.2]
    for i,min_overlap in enumerate(min_overlap_vals):
        model.min_overlap = min_overlap
        metrics = get_metrics(
            metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
            N=20,
            model=model,
            train_user_items=train_plays.transpose(),
            test_user_items=test_plays.transpose(),
            song_df=song_df,
            limit=limit_users)
        MAPk_vals[i] = metrics["MAP@K"]
        cos_dissim_vals[i] = metrics["mean_cosine_list_dissimilarity"]
        print(metrics)
        
    model.min_overlap = 0.05
    all_MAPk_vals.append(MAPk_vals)
    all_cos_dissim_vals.append(cos_dissim_vals)
    
    print(MAPk_vals)
    plt.figure()
    plt.title("MAP@K for varying minimum MUSIC profile overlap")
    plt.xlabel("Minimum MUSIC profile overlap")
    plt.ylabel("MAP@K value")
    plt.plot(min_overlap_vals,MAPk_vals)
#     plt.show(block=False)
    plt.savefig("./figures/mapk_overlap")

    
    print(cos_dissim_vals)
    plt.figure()
    plt.title("Cosine list dissimilarity for varying minimum MUSIC profile overlap")
    plt.xlabel("Minimum MUSIC profile overlap")
    plt.ylabel("List Dissimilarity")
    plt.plot(min_overlap_vals,cos_dissim_vals)
#     plt.show(block=False)
    plt.savefig("./figures/div_overlap")
    
    #redefine
    MAPk_vals = np.zeros(NUM_VALS)
    cos_dissim_vals = np.zeros(NUM_VALS)
    k_vals = [10,50,100,500,1000]
    for i,k_val in enumerate(k_vals):
        model.k = k_val
        metrics = get_metrics(
            metrics=['MAP@K', 'mean_cosine_list_dissimilarity'],
            N=20,
            model=model,
            train_user_items=train_plays.transpose(),
            test_user_items=test_plays.transpose(),
            song_df=song_df,
            limit=limit_users)
        MAPk_vals[i] = metrics["MAP@K"]
        cos_dissim_vals[i] = metrics["mean_cosine_list_dissimilarity"]
        print(metrics)
        
    model.k = 100
    all_MAPk_vals.append(MAPk_vals)
    all_cos_dissim_vals.append(cos_dissim_vals)
    
    
    print(MAPk_vals)
    plt.figure()
    plt.title("MAP@K for number of kNN neighbours")
    plt.xlabel("Number of kNN neighbours")
    plt.ylabel("MAP@K value")
    plt.plot(k_vals,MAPk_vals)
#     plt.show(block=False)
    plt.savefig("./figures/mapk_knn")

    print(cos_dissim_vals)
    plt.figure()
    plt.title("Cosine list dissimilarity for varying number of kNN neighbours")
    plt.xlabel("Number of kNN neighbours")
    plt.ylabel("List Dissimilarity")
    plt.plot(k_vals,cos_dissim_vals)
#     plt.show(block=False)
    plt.savefig("./figures/div_knn")
    
    np.save("./data/MAPk_vals",np.array(all_MAPk_vals))
    np.save("./data/cos_dissim",np.array(all_cos_dissim_vals))

