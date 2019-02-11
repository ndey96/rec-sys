# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:35:09 2019

@author: Matthew
"""

import os
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import codecs
import logging
import time
import tqdm

import numpy as np


from implicit.als import AlternatingLeastSquares
#from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
#                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)
from implicit.datasets.lastfm import get_lastfm
from implicit.datasets.million_song_dataset import get_msd_taste_profile

from implicit.evaluation import train_test_split
from implicit.evaluation import mean_average_precision_at_k

class RandomRecommender:
    def __init__(self,min_bound,high_bound,confidence):
        self.min_bound = min_bound
        self.high_bound = high_bound
        self.confidence = confidence
    
    def recommend(self,userid,train_csr,N):
        recs = np.random.randint(self.min_bound,self.high_bound,N)
        recs = [(rec,self.confidence) for rec in recs]
        return recs
    
    def fit(self,train_data):
        return 1

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
#          "nmslib_als": NMSLibAlternatingLeastSquares,
#          "annoy_als": AnnoyAlternatingLeastSquares,
#          "faiss_als": FaissAlternatingLeastSquares,
#          "tfidf": TFIDFRecommender,
#          "cosine": CosineRecommender,
          "bpr": BayesianPersonalizedRanking,
          "bm25": BM25Recommender,
          "random":RandomRecommender}


def get_model(model_name):
    print("getting model %s" % model_name)
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 16, 'dtype': np.float32}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    elif model_name == "bpr":
        params = {'factors': 63}
    else:
        params = {}

    return model_class(**params)

if __name__ == "__main__":
    songs, users, plays = get_msd_taste_profile()
    train_plays, test_plays = train_test_split(plays)
    
    model_name = 'als'
    num_users_to_recommend = 1000
    num_recommendations = 100
    model = get_model(model_name)
    
    if issubclass(model.__class__, AlternatingLeastSquares):
        bm25_play = bm25_weight(plays, K1=100, B=0.8)
        train_plays, test_plays = train_test_split(bm25_play)
            # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")

            # also disable building approximate recommend index
#            model.approximate_similar_items = False
    else:
        train_plays, test_plays = train_test_split(plays)
        #%%        
    model.fit(train_plays)

    #%%
    complete_recs = []
    embedding_mapping = []
    for i in range(num_users_to_recommend):
        recommendations = model.recommend(i,train_plays,num_recommendations)
        recs_with_songid = [(songs[rec[0]][0],users[i]) for rec in recommendations]
        complete_recs.append(recs_with_songid)
        
    for song, embedding in zip(songs, model.item_factors):
        embedding_mapping.append([song,embedding])
    
    embedding_mapping = np.array(embedding_mapping)
    complete_recs = np.array(complete_recs)
    np.save("recommendations",complete_recs)
    np.save('embeddings',embedding_mapping)
    
        
        