# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:56:28 2018

@author: Matthew
"""

from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold



import pandas as pd

PATH_TO_DATA = '../music/train_triplets.csv'
data_df  = pd.read_csv(PATH_TO_DATA,sep='\t',names=['UserID','SongID','NumPlays'], nrows=10000)

data_df['Rating'] = 1
reader = Reader(rating_scale=(0,1))
data = Dataset.load_from_df(data_df[['UserID','SongID','Rating']],reader)

algo = SVD()

kf = KFold(n_splits=2)

for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)