# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:31:37 2018

@author: Matthew
"""
import pandas as pd


PATH_TO_USAGE = '../data/train_triplets.csv'
PATH_TO_SONG_TO_TRACK = '../data/taste_profile_song_to_tracks.txt'


#Uncomment when you need to load data for first time if you are using an IDE like Spyder (efficiency reason)
usage_data  = pd.read_csv(PATH_TO_USAGE,sep='\t',names=['UserID','SongID','NumPlays'])
song_to_track = pd.read_csv(PATH_TO_SONG_TO_TRACK, sep='\t', names =['SongID','TrackID'])

usage_data_with_track = pd.merge(usage_data, song_to_track, on='SongID',how='inner')
track_genres = pd.read_table('http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-MAGD-genreAssignment.cls',names=['TrackID','Genre'],sep='\t')

data_with_genre = pd.merge(usage_data_with_track,track_genres,on='TrackID',how='inner')

data_with_genre.to_csv("tracks_with_genre")