# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:25:56 2019

@author: Matthew McLeod
"""

import pandas as pd
import numpy as np
import h5py


from implicit.datasets.million_song_dataset import _read_triplets_dataframe
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz
#%%
#Load the data
# The partial history for train and test is available at:  https://labrosa.ee.columbia.edu/millionsong/challenge#data1
# The base triplets is available at: https://labrosa.ee.columbia.edu/millionsong/tasteprofile

#NOTE: Modify library code to not be hardcoded in the _read_... function
train_base_path = './Data/train_triplets.txt'
train_test_path = './Data/EvalDataYear1MSDWebsite/year1_test_triplets_visible.txt'
test_path = './Data/EvalDataYear1MSDWebsite/year1_test_triplets_hidden.txt'

print("Reading in Data")
train_partial_data = _read_triplets_dataframe(train_test_path)
train_full_data = _read_triplets_dataframe(train_base_path)
test_partial_data = _read_triplets_dataframe(test_path)

#%%
#Get the full dataset since I will need to put this in COO matrix. It will be difficult matching up the song ids of test and train
#if they are treated independently. Putting them together and then splitting the train and test set when creating the COO matrix is the
#easiest way of ensuring that the indices remain correct.
print("Aggregating Data")
data = test_partial_data.append(train_partial_data).append(train_full_data)
data['user'] = data['user'].astype("category")
data['track'] = data['track'].astype("category")

#%%
#Build the train and test CSR matrices. 
print("Building CSR matrix")
#Since test_partial_data was the first dataframe (with the others appended), the first n entries for the csr matrix will then be the first n entries of the cat codes
test_index = test_partial_data.shape[0]

num_users = data['user'].cat.codes.unique().shape[0]
num_songs = data['track'].cat.codes.unique().shape[0]

#take the first entries which would below to test_partial_data
rows_test = data['track'].cat.codes.copy()[:test_index]
cols_test = data['user'].cat.codes.copy()[:test_index]
data_test = data['plays'].astype(np.float32)[:test_index]

#The rest is train
rows_train = data['track'].cat.codes.copy()[test_index:]
cols_train = data['user'].cat.codes.copy()[test_index:]
data_train = data['plays'].astype(np.float32)[test_index:]

#not sure, but I think it is important to specify that these objects will be the same shape
test_plays = coo_matrix((data_test,(rows_test,cols_test)),shape=(num_songs,num_users)).tocsr()
train_plays = coo_matrix((data_train,(rows_train,cols_train)),shape=(num_songs,num_users)).tocsr()

#%%
#Save to file
train_filename = 'train_sparse'
test_filename = 'test_sparse'
user_mapping_filename = 'user_mapping.csv'
song_mapping_filename = 'song_mapping.csv'

user_mapping = np.vstack((data['user'].cat.codes.copy().values,data['user'].values)).T
song_mapping = np.vstack((data['track'].cat.codes.copy().values,data['track'].values)).T

user_id_to_user_index = pd.DataFrame(columns=['user','sparse_index'],data=user_mapping)
song_id_to_song_index = pd.DataFrame(columns=['track','sparse_index'],data=song_mapping)
#Should be moved to hdf5 format since csv takes long time
user_id_to_user_index.to_csv(user_mapping_filename)
song_id_to_song_index.to_csv(song_mapping_filename)

save_npz(train_filename,train_plays)
save_npz(test_filename,test_plays)

