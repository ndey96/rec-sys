# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:25:45 2018

@author: Ian
"""

import implicit
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

model = implicit.als.AlternatingLeastSquares(factors=50)

data_df = pd.read_csv('data/triplets.csv')[['user_id', 'song_id']]
user_item = data_df.groupby(['user_id', 'song_id']).size().unstack(fill_value=0)
user_item_csr = csr_matrix(user_item)
model.fit(user_item_csr)

#silly, but users aren't referenced by their "user_id" instead it is their actual index
test_user_id = 0

#top recommendations
recommendations = model.recommend(test_user_id, user_item_csr)
#top similar users
similar_users = model.similar_users(test_user_id)
#to get the broken down item factors, you can access the model attribute
song_embedding = model.item_factors