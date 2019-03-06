from tqdm import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx


def py_mean_average_precision_at_k(model,
                                   train_user_items,
                                   test_user_items,
                                   K=10):
    """ Calculates MAP@K for a given trained model
    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        csr matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        csr matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    Returns
    -------
    float
        the calculated MAP@k
    """

    

    user_items_coo = test_user_items.tocoo()
    user_to_listened_songs_map = {}
    for user_index, song_index in zip(user_items_coo.row, user_items_coo.col):
        if user_index not in user_to_listened_songs_map:
            user_to_listened_songs_map[user_index] = set()
        user_to_listened_songs_map[user_index].add(song_index)


    average_precision_sum = 0
    for user_index, listened_song_indices in tqdm(user_to_listened_songs_map.items()):
        recommended_song_indices = [
            rec[0]
            for rec in model.recommend(user_index, train_user_items, N=K)
        ]

        # http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
        precision_sum = 0
        for k in range(1, K):
            k_recommended_song_indices = set(recommended_song_indices[:k])
            num_correct_recs = len(listened_song_indices & k_recommended_song_indices)
            precision = num_correct_recs / k
            precision_sum += precision

        average_precision = precision_sum / min(K, len(listened_song_indices))
        average_precision_sum += average_precision
        # average_precision = 0
        # relevant = 0
        #     if likes.find(song_ids[i]) != likes.end():
        #         relevant += 1
        #         average_precision += relevant / (i + 1)

        # mean_ap += average_precision / min(K, len(likes))
        # total += 1

    num_test_users = user_to_listened_songs_map
    mean_average_precision = average_precision_sum / total_num_users
    return mean_average_precision


if __name__ == '__main__':
    from ALSpkNN import get_baseline_cf_model, weight_cf_matrix
    from scipy.sparse import load_npz
    import time

    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')

    # print("Building and fitting the baseline CF model")
    baseline_cf_model = get_baseline_cf_model()
    # weighted_train_csr = weight_cf_matrix(train_plays, alpha=1)
    baseline_cf_model.fit(train_plays)

    # print("Evaluating the baseline CF model")
    start = time.time()

    # setting num_threads = 0 yields a 3x speedup on Nolan's MBP
    MAPk = py_mean_average_precision_at_k(
        model=baseline_cf_model,
        train_user_items=train_plays.transpose(),
        test_user_items=test_plays.transpose(),
        K=5)

    print("MAPK for baseline CF is: " + str(MAPk))
    print(f'Calculation took {time.time() - start}s')

