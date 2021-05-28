# Author: Cenk Tüysüz
# Date: 16.03.2020
# Script to divide data into 2 folders

# Instructions

# Change in_loc and out_loc
# Change n_subgraphs
# Change ratio
# then execute

import os
from random import shuffle
from shutil import copyfile
import numpy as np

in_loc  = 'data/graph_data/mu200_1pT/hitgraphs/'
out_loc = 'data/graph_data/mu200_1pT/dataset_50_50/'

print('Reading data from: ' + in_loc)

dirs = os.listdir(out_loc)
if 'train' in dirs:
    t = os.listdir(out_loc+'train/')
    for file in t:
        os.remove(out_loc+'train/'+file)
else:
    os.mkdir(out_loc+'train')
if 'valid' in dirs:
    v = os.listdir(out_loc+'valid/')
    for file in v:
        os.remove(out_loc+'valid/'+file)
else:
    os.mkdir(out_loc+'valid')

n_subgraphs = 1

files = sorted(os.listdir(in_loc))
print('Found %d files.'%len(files))
n_event = len(files)//n_subgraphs
print('Found %d events.'%n_event)

ratio = 0.5
event_list = np.arange(n_event)

shuffle(event_list)

n_train = int(n_event*ratio)
n_valid = n_event - n_train

train_list = event_list[:n_train]
valid_list = event_list[n_train:n_train+n_valid]

for event_id in train_list:
    for subgraph in range(n_subgraphs):
        copyfile(in_loc+files[subgraph+event_id*n_subgraphs],out_loc+'train/'+files[subgraph+event_id*n_subgraphs])
for event_id in valid_list:
    for subgraph in range(n_subgraphs):
        copyfile(in_loc+files[subgraph+event_id*n_subgraphs],out_loc+'valid/'+files[subgraph+event_id*n_subgraphs])

t_files = os.listdir(out_loc+'train')
v_files = os.listdir(out_loc+'valid')
print(str(len(t_files)) + ' files copied to ' + out_loc+'train/')
print(str(len(v_files)) + ' files copied to ' + out_loc+'valid/')
print('Divided the dataset of ' + str(n_event) + ' events with ratio ' + str(len(t_files)/len(v_files)) +' to 1')
