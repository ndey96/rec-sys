songs_df = pd.read_hdf('data/full_msd_with_audio_features.h5', key='df')

def get_diversity(song_vector_list):
    dissim = 0
    n = len(song_vector_list)
    
    for i in range(n):
        for j in range(n):
            dissim += np.linalg.norm(song_vector_list[i] - song_vector_list[j])
            
    return dissim/((n/2)*(n-1))

def convert_recs_to_embeddings(rec_list):
    embeddings_list = []
    
    for rec_ids in rec_list:
        #print(prepare_song_id(rec_ids[0]))
        if (prepare_song_id(rec_ids[0]) in songs_df.song_id.values):
            row = songs_df.loc[songs_df['song_id'] == prepare_song_id(rec_ids[0])]
            embedding = row.iloc[:,12:]
            embedding = np.array(embedding.values.tolist()[0])
            embeddings_list.append(embedding)
        
    return embeddings_list

def get_diversity_score(rec_list):
    # convert to list of embeddings
    embedding_list = convert_recs_to_embeddings(rec_list)
    
    # calculate diversity
    return get_diversity(embedding_list)
    
    