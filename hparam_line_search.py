import matplotlib.pyplot as plt
from evaluation_hyperopt import get_metrics
import pandas as pd
from scipy.sparse import load_npz
from ALSpkNN import ALSpkNN
import numpy as np
import json
import time
import sys
sys.setrecursionlimit(10000)


def assign_defaults(default_vals, model):
    for key in list(default_vals.keys()):
        setattr(model, key, default_vals[key])


if __name__ == '__main__':

    print('Loading Data...')
    train_plays = load_npz('data/train_sparse.npz')
    test_plays = load_npz('data/test_sparse.npz')
    song_df = pd.read_hdf('data/song_df.h5', key='df')
    user_df = pd.read_hdf('data/user_df.h5', key='df')


    #order of placement in dictionary matters
    hparam_vals_map = {
        'k': [50, 100, 200, 500],
        'max_overlap': [0.05, 0.1, 0.15, 0.2],
        'knn_frac': [0, 0.25, 0.5, 1],
        'min_songs': [1, 5, 10, 15],
    }

    default_vals = {
        'k': 100,
        'max_overlap': 0.1,
        'knn_frac': 0.5,
        'min_songs': 5,
        'cf_weighting_alpha': 1,
    }

    metrics_to_get = [
        'MAP@K', 'mean_cosine_list_dissimilarity', 'metadata_diversity'
    ]


    USER_LIMIT = 10000
    for hparam_name, hparam_vals in hparam_vals_map.items():
        print(f'\n\nStarting {hparam_name} line search...')
        start = time.time()
        results = np.zeros((len(metrics_to_get), len(hparam_vals)))

        for i, val in enumerate(hparam_vals):
            model_params = {**default_vals, hparam_name: val}
            print(f'Building model with {model_params}')
            model = ALSpkNN(user_df, song_df, **model_params)
            print('Fitting model...')
            model.fit(train_plays)

            metrics = get_metrics(
                metrics=metrics_to_get,
                N=20,
                model=model,
                train_user_items=train_plays.transpose(),
                test_user_items=test_plays.transpose(),
                song_df=song_df,
                limit=USER_LIMIT)
            print(metrics)
            results[:, i] = list(metrics.values())

        for i in range(len(metrics_to_get)):
            metric_name = metrics_to_get[i]
            plt.figure()
            plt.title(f'Effect of varying {hparam_name}')
            plt.plot(hparam_vals, results[i, :])
            plt.ylabel(metric_name)
            plt.xlabel(hparam_name + ' values')
            plt.tight_layout()
            plt.savefig(f'figures_10k_N_20/{hparam_name}_{metric_name}')
            np.save(f'figures_10k_N_20/results_{hparam_name}', results)

        print(f'{hparam_name} line search took {time.time()-start}s')
