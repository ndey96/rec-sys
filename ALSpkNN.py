# -*- coding: utf-8 -*-

from implicit.als import AlternatingLeastSquares
import os
os.environ['MKL_NUM_THREADS'] = '1'
from scipy.spatial import KDTree
import numpy as np
from collections import Counter
import utilities

def cf_weighting(csr_mat):
    alpha = 1
    #don't want to modify original incase it gets put into other models
    new_csr = csr_mat.copy()
    new_csr.data = 1 + np.log(alpha*csr_mat.data)
    return new_csr

class ALSpkNN():
    # building the knn should probably be in the fit method as well... but it doesn't really matter
    def __init__(self,user_MUSIC_df,user_mapping,song_mapping,user_playlist_df):
        #build the collaborative filtering model with params hardcoded
        self._build_cf_model()
        
        #build the knn tree
        self._build_kdtree(user_MUSIC_df)
        self._user_MUSIC_df = user_MUSIC_df
        self._user_mapping = user_mapping
        self._song_mapping = song_mapping
        self._user_playlist_df = user_playlist_df
    
    def _build_kdtree(self,user_MUSIC_df):
        MUSIC_vectors = user_MUSIC_df['MUSIC'].values.tolist()
        self.kdtree = KDTree(MUSIC_vectors)
    
    def _build_cf_model(self):
        als_params = {'factors': 16,
                  'dtype': np.float32,
                  'iterations': 2,
                  'calculate_training_loss': True}
        model = AlternatingLeastSquares(**als_params)

        #perform operation to approximate confidence intervals
        #paper doesn't specify an alpha value, so just guess alpha=1

        self.cf_model = model
    
    def fit(self, train_csr):
        modified_train_csr = cf_weighting(train_csr)
        self.cf_model.fit(modified_train_csr)
        
    #note: userid is not the csr user id
    '''
       #            # mODIFIED CODE TO BE .ILOC[...,2] INSTEAD OF 3
#             I DON'T KNOW HOW THIS WAS ABLE TO RUN FOR YOU GUYS SINCE USER_MUSIC_DF
#            ONLY HAS 3 COLUMNS, SO THEREFORE, THE MAX RANGE OF ILOC IS 2
#            closest_userids.append(self._user_MUSIC_df.iloc[index, 3])
    '''
    def _get_closest_MUSIC_user_ids(self, user_id, k, user_MUSIC_df, kdtree):
        user_MUSIC = self._user_MUSIC_df.loc[self._user_MUSIC_df['user_id'] == user_id]['MUSIC'].tolist()[0]

        distances, indices = kdtree.query(user_MUSIC, k)
        closest_userids = []

        for index in indices:
            closest_userids.append(self._user_MUSIC_df.iloc[index, 2])
        return closest_userids
 
    
    
    
    #note: userid is not the csr user id
    def get_knn_top_m_songs(self, user_id, k, m, kdtree, user_playlist_df, user_MUSIC_df):
        closest_userids = self._get_closest_MUSIC_user_ids(user_id, k, user_MUSIC_df, kdtree)
        closest_user_songs = []
        for i in range(len(closest_userids)):
            closest_user_songs.append(user_playlist_df.loc[user_playlist_df[
                    'user_id'] == closest_userids[i]]['playlist'].tolist()[0])

        closest_user_songs = [item for sublist in closest_user_songs for item in sublist]
        counted_closest_user_songs = Counter(closest_user_songs)
        top_m_songs = [i[0] for i in counted_closest_user_songs.most_common()[:m]]

        return top_m_songs
    
    #user_id, user_sparse_index, cf_model, kdtree, train_plays, user_playlist_df, user_MUSIC_df, n, m, k

    def recommend(self,user_sparse_index,train_plays,N):
        n_songs = [song_tuple[0] for song_tuple in self.cf_model.recommend(userid=user_sparse_index,
                                                                      user_items=train_plays.transpose(),
                                                                      N=N)]
    
        #will need to use mapping dic
        true_user_id = self._user_mapping.loc[self._user_mapping.sparse_index == user_sparse_index].user.values[0]
        
        #todo Fix 
        #We want to modify the ratio of n and m songs.
        k=N
        m_songs = self.get_knn_top_m_songs(user_id=true_user_id,
                                      k=k,
                                      m=N,
                                      kdtree=self.kdtree,
                                      user_playlist_df=self._user_playlist_df,
                                      user_MUSIC_df=self._user_MUSIC_df)
        
        #may need to map m_songs to internal song id
        #I don't think this is very efficient for looking up
        m_songs = [self._song_mapping.loc[self._song_mapping.track == song].sparse_index.values[0] for song in m_songs]
        
        rec_list =  utilities.concat_shuffle(n_songs,m_songs)
        return rec_list[:N]