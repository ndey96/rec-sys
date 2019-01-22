# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:15:12 2019

@author: Matthew
"""

import json
import os
import pandas as pd

base_path = 'Data/millionsongdataset_echonest/'
folder_list = os.listdir(base_path)

mapping = pd.Series()

for folder_name in folder_list:
    file_list = os.listdir(base_path+folder_name)
    print("Starting Directory : " + str(folder_name))
    for filename in file_list:
        filename = base_path + folder_name + "/" + filename
#        print(filename)
        with open(filename) as f:
            data_all = json.load(f)['response']['songs']
            if (len(data_all) == 0):
                # this is the wierd and rare case that a song on MSD has not matches on spotify
                continue
            if (len(data_all) > 1):
                print("HEY! WATCHOUT! THIS IS A CASE YOU NEVER SAW. WHAT INFO IS HERE")
            
            data = data_all[0]
            
            MSDsongid = data['id']
            MSDartistid = data['artist_id']
            for trackPackage in data['tracks']:
                if trackPackage['catalog'] == 'spotify':
                    spotifyId = trackPackage['foreign_id'].split(":")[2]
                    mapping[spotifyId] = MSDsongid
        break
        
        
        
    