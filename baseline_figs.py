import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)

figure_dir = 'figures_91k_N_20-apr-17'
hparam_vals_map = {
    'mode': ['popular', 'weighted_random', 'random'],
    'k': [20, 30, 50, 75, 100, 250, 500, 750, 1000],
    'knn_frac': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}

# hparam_vals_map = {
#     'mode': ['popular', 'weighted_random', 'random'],
#     'k': [20, 30, 50, 75, 100, 250, 500, 750, 1000],
#     'knn_frac': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#     'max_overlap': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
# }
# metrics_to_get = ['MAP@K', 'num_genres']
metrics_to_get = ['MAP@K', 'mean_cosine_list_dissimilarity']

fig, axes = plt.subplots(len(metrics_to_get), len(hparam_vals_map), figsize=(20,10))
for i in range(len(metrics_to_get)):
    ax_row = axes[i]
    metric_name = metrics_to_get[i]
    ax_row[0].set_ylabel(metric_name)

    for (hparam_name, hparam_vals), ax in zip(hparam_vals_map.items(), ax_row):
        results = np.load(f'{figure_dir}/results_{hparam_name}.npy')
#         ax.set_title(f'Effect of varying {hparam_name}')
        ax.plot(hparam_vals, results[i, :], linestyle='--', marker='o', color='black')
        
        ax.set_xlabel(hparam_name)
        
        xmin = hparam_vals[0]
        xmax = hparam_vals[len(hparam_vals)-1]

        if metric_name == 'MAP@K':
            ax.hlines(0.10620597746944373,xmin, xmax, color='blue',linestyles='dashed', label='Popular')
            ax.hlines(0.0001349077479153938,xmin, xmax, color='green', linestyles='dashed', label='Random')
            ax.hlines(0.0855785767079789,xmin, xmax, color='orange',linestyles='dashed', label='ALS')
            ax.hlines(0.0059247763602836324,xmin, xmax, color='red',linestyles='dashed', label='Weighted Random')

        if metric_name == 'mean_cosine_list_dissimilarity':
            ax.hlines(0.1782422452673025,xmin, xmax,color='blue',linestyles='dashed', label='Popular')
            ax.hlines(0.17520191979126418,xmin, xmax,color='green', linestyles='dashed', label='Random')
            ax.hlines(0.16196735253426256,xmin, xmax, color='orange',linestyles='dashed', label='ALS')
            ax.hlines(0.15991899935008733,xmin, xmax, color='red',linestyles='dashed', label='Weighted Random')

        if metric_name == 'metadata_diversity':
            ax.hlines(2.708892926008163,xmin, xmax,color='blue', linestyles='dashed', label='Popular')
            ax.hlines(4.511948377743507,xmin, xmax,color='green', linestyles='dashed', label='Random')
            ax.hlines(3.223943220517218,xmin, xmax,color='orange', linestyles='dashed', label='ALS')
            ax.hlines(4.121585381415222,xmin, xmax, color='red',linestyles='dashed', label='Weighted Random')

        if metric_name == 'num_genres':
            ax.hlines(1.9996506626492871,xmin, xmax,color='blue', linestyles='dashed', label='Popular')
            ax.hlines(4.17823846640903,xmin, xmax,color='green', linestyles='dashed', label='Random')
            ax.hlines(3.0329796292657365,xmin, xmax,color='orange', linestyles='dashed', label='ALS')
            ax.hlines(3.7169166612082707,xmin, xmax, color='red',linestyles='dashed', label='Weighted Random')

        ax.legend()

plt.tight_layout()
plt.savefig('paper_graphs.png')