# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 12:29:41 2019

@author: Matthew
"""

""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/
This code will automically download a HDF5 version of the dataset from
GitHub when it is first run. The original dataset can also be found at
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html.
"""
import os
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import codecs
import logging
import time
import tqdm

import implicit.cuda

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
    
class AlternatingLeastSquaresDiversity(AlternatingLeastSquares):

    def __init__(self, factors=100, regularization=0.01, dtype=np.float32,
                 use_native=True, use_cg=True, use_gpu=implicit.cuda.HAS_CUDA,
                 iterations=15, calculate_training_loss=False, num_threads=0):
         # use the base ALS initialization   
        super().__init__(factors, regularization,dtype,use_native,use_cg,use_gpu,iterations,calculate_training_loss,num_threads)
        
    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False):
        recs = super().recommend(userid,user_items,N,filter_already_liked_items,filter_items,recalculate_user)
        recs = self.diversity_function(recs)
        return recs
        
    def diversity_function(self,recs):
        return recs

# maps command line model argument to class name
MODELS = {
#        'als_diversity':AlternatingLeastSquaresDiversity,
        "als":  AlternatingLeastSquares,

#          "nmslib_als": NMSLibAlternatingLeastSquares,
#          "annoy_als": AnnoyAlternatingLeastSquares,
#          "faiss_als": FaissAlternatingLeastSquares,
#          "tfidf": TFIDFRecommender,
#          "cosine": CosineRecommender,
#          "bpr": BayesianPersonalizedRanking,
#          "bm25": BM25Recommender,
#          "random":RandomRecommender
}


def get_model(model_name):
    print("getting model %s" % model_name)
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 16, 'dtype': np.float32,'iterations':15,'calculate_training_loss':True}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    elif model_name == "bpr":
        params = {'factors': 63}
    else:
        params = {}

    return model_class(**params)

if __name__ == "__main__":
    songs, users, plays = get_msd_taste_profile()
#%%
    
    MAPk_scores = []
    for model_name in MODELS:
        
        model = get_model(model_name)
#        model = RandomRecommender(1,len(songs),1)
            # if we're training an ALS based model, weight input for last.fm
            # by bm25
        if issubclass(model.__class__, AlternatingLeastSquares):
            bm25_play = bm25_weight(plays, K1=100, B=0.8)
            train_plays, test_plays = train_test_split(bm25_play)
            # lets weight these models by bm25weight.
            logging.debug("weighting matrix by bm25_weight")

            # also disable building approximate recommend index
#            model.approximate_similar_items = False
        else:
            train_plays, test_plays = train_test_split(plays)

        model.fit(train_plays)
        MAPk = mean_average_precision_at_k(model,train_plays,test_plays,K=10)
        MAPk_scores.append(MAPk)
#        break
        
        
        print("MAP for "+str(model_name)+" is: "+ str(MAPk))

