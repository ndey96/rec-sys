import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from metrics import get_metrics
from models import ALSpkNN, ALSRecommender, PopularRecommender, RandomRecommender, WeightedRecommender

train_plays = load_npz('data/train_sparse.npz')
test_plays = load_npz('data/test_sparse.npz')
song_df = pd.read_hdf('data/song_df.h5', key='df')
user_df = pd.read_hdf('data/user_df.h5', key='df')
metric_list = ['MAP@K', 'mean_cosine_list_dissimilarity', 'metadata_diversity']
user_limit = 10#999999
results = {}

models = {
    'popular': PopularRecommender,
    'random': RandomRecommender,
    'weighted': WeightedRecommender,
    'ALS': ALSRecommender,
}
for model_name, model_class in models.items():
    print(f'\n\nCalculating metrics for {model_name}')
    m = model_class()
    m.fit(train_plays)
    results[model_name] = get_metrics(
        metrics=metric_list,
        N=20, 
        model=m,
        train_user_items=train_plays.transpose(),
        test_user_items=test_plays.transpose(),
        song_df=song_df,
        limit=user_limit)
print(results)

with open(f'baseline_metrics_user_limit_{user_limit}.json', 'w') as fp:
    json.dump(results, fp, indent=4)