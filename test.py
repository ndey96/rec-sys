import numpy as np
import matplotlib.pyplot as plt

metrics_to_get = ['MAP@K', 'mean_cosine_list_dissimilarity', 'metadata_diversity']
hparams = ['mode', 'knn_frac', 'k']
figure_dir = 'figures_91k_N_20'


hparam_vals_map = {
    'mode': ['popular', 'weighted_random', 'random'],
    'knn_frac': [0.25, 0.5, 0.75, 1],
    'k': [50, 100, 250, 500],
}

for hparam_name, hparam_vals in hparam_vals_map.items():
    results = np.load(f'{figure_dir}/results_{hparam_name}.npy')
    for i in range(len(metrics_to_get)):
        metric_name = metrics_to_get[i]
        plt.figure()
        plt.title(f'Effect of varying {hparam_name}')
        plt.plot(hparam_vals, results[i, :], linestyle='--', marker='o', color='b')

        xmin = hparam_vals[0]
        xmax = hparam_vals[len(hparam_vals)-1]

        if metric_name == 'MAP@K':
            plt.hlines(0.10620597746944373,xmin, xmax, color='red',linestyles='dashed', label='popular')
            plt.hlines(0.0001349077479153938,xmin, xmax, color='green', linestyles='dashed', label='random')
            plt.hlines(0.0855785767079789,xmin, xmax, color='black',linestyles='dashed', label='ALS')

        if metric_name == 'mean_cosine_list_dissimilarity':
            plt.hlines(0.1782422452673025,xmin, xmax,color='red',linestyles='dashed', label='popular')
            plt.hlines(0.17520191979126418,xmin, xmax,color='green', linestyles='dashed', label='random')
            plt.hlines(0.16196735253426256,xmin, xmax, color='black',linestyles='dashed', label='ALS')

        if metric_name == 'metadata_diversity':
            plt.hlines(2.708892926008163,xmin, xmax,color='red', linestyles='dashed', label='popular')
            plt.hlines(4.511948377743507,xmin, xmax,color='green', linestyles='dashed', label='random')
            plt.hlines(3.223943220517218,xmin, xmax,color='black', linestyles='dashed', label='ALS')

        plt.ylabel(metric_name)
        plt.xlabel(hparam_name + ' values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'baseline_figs/{hparam_name}_{metric_name}')