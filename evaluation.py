from tqdm import trange
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# import cython
# from cython.operator import dereference
# from cython.parallel import parallel, prange
# from libc.stdlib cimport malloc, free
# from libc.string cimport memset
# from libc.math cimport fmin
# from libcpp.unordered_set cimport unordered_set

# https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx


def py_mean_average_precision_at_k(model,
                                   train_user_items,
                                   test_user_items,
                                   K=10,
                                   show_progress=True,
                                   num_threads=1):
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
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.
    Returns
    -------
    float
        the calculated MAP@k
    """

    total = 0
    total_num_users = test_user_items.shape[0]
    test_indptr = test_user_items.indptr

    # Supposed to represent song_indices I think?
    # Should be user indices in practice though
    test_indices = test_user_items.indices

    average_precision_sum = 0
    for u in trange(total_num_users):
        # if we don't have any test items, skip this user
        if test_indptr[u] == test_indptr[u + 1]:
            continue

        # recommended_song_indices -> the set of songs that were recommended
        recommended_song_indices = {
            rec[0]
            for rec in model.recommend(u, train_user_items, N=K)
        }

        # listened_song_indices -> the set of songs that user has listened to
        listened_song_indices = {
            test_indices[i]
            for i in range(test_indptr[u], test_indptr[u + 1])
        }

        # http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
        precision_sum = 0
        for k in range(1, K):
            num_recs_in_listening_history = len(
                listened_song_indices & recommended_song_indices)
            precision = num_recs_in_listening_history / k
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
        K=5,
        show_progress=True,
        num_threads=0)

    print("MAPK for baseline CF is: " + str(MAPk))
    print(f'Calculation took {time.time() - start}s')

