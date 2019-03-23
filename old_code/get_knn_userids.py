import pandas as pd
from scipy.spatial import KDTree
import sys

def build_tree():
    
    user_MUSIC_df = pd.read_hdf('data/user_div_MUSIC_divpref.h5', key='df')
    MUSIC_vectors = user_MUSIC_df.iloc[:,0].values.tolist()
    tree = KDTree(MUSIC_vectors)
    
    return tree

def get_closest_MUSIC_users(userid, k):

    user_MUSIC = user_MUSIC_df.loc[user_MUSIC_df['user_id'] == userid]['MUSIC'].tolist()[0]
    
    distances, indices = tree.query(user_MUSIC, k)
    closest_userids = []
    
    for index in indices:
        closest_userids.append(user_MUSIC_df.iloc[index, 3])
        
    print(closest_userids)
    
    return closest_userids

if __name__ == "__main__":
    get_closest_MUSIC_users(sys.argv[1], int(sys.argv[2]))

    