import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
from tools.tools import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
import hdbscan

def find_hit_id(hit):
	for idx, hitx in enumerate(X):
		if (hit==hitx).all():
			return idx

filename = 'data/graph_data/mu10_big/train_200/valid/event000000009_g000.npz'

X, Ri, Ro, y = load_graph(filename)

print('Number of hits: %d'%X.shape[0])
print('Number of edges: %d'%y.shape[0])

with open('logs/test/log_validation_preds.csv', 'r') as f:
	reader = csv.reader(f, delimiter=',')  
	preds = np.array(list(reader)).astype(float)
	preds = preds[-y.shape[0]:,:]
	

bo = np.transpose(Ro) @ X
bi = np.transpose(Ri) @ X


eps = 0.7
min_samples = 2

e = np.ones((X.shape[0],X.shape[0]))

for idx, hit in enumerate(bo):
	hid_bo =find_hit_id(bo[idx])
	hid_bi =find_hit_id(bi[idx])
	e[hid_bo,hid_bo] = 0
	e[hid_bi,hid_bi] = 0
	if preds[idx,0] > eps:
		e[hid_bo,hid_bi] = 0
		e[hid_bi,hid_bo] = 0




#db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(e)
#labels = db.labels_

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
labels = clusterer.fit(e).labels_


e_ = np.ones((X.shape[0],X.shape[0]))

for idx, hit in enumerate(bo):
	hid_bo =find_hit_id(bo[idx])
	hid_bi =find_hit_id(bi[idx])
	e_[hid_bo,hid_bo] = 0
	e_[hid_bi,hid_bi] = 0
	e_[hid_bo,hid_bi] = 1 - y[idx]
	e_[hid_bi,hid_bo] = 1 - y[idx]

#db_tr = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(e_)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
labels_tr = clusterer.fit(e_).labels_

#labels_tr = db_tr.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('From predictions:')
print('Estimated number of tracks: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_tr)) - (1 if -1 in labels_tr else 0)
n_noise_ = list(labels_tr).count(-1)
print('From truth data:')
print('Estimated number of tracks: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


fig, ax = plt.subplots(1,3,figsize = (16,5),sharey=True, tight_layout=True)
#print('Zmin: %.2f, Zmax: %.2f' %(min(X[:,2]),max(X[:,2]))   )
X_ = np.copy(X)
X_[:,1] = X[:,1] * np.pi
theta = (X_[:,1])%(np.pi*2)
ax[0].scatter(1000*X_[:,0]*np.cos(theta), 1000*X_[:,0]*np.sin(theta), c='k', s=3)
ax[1].scatter(1000*X_[:,0]*np.cos(theta), 1000*X_[:,0]*np.sin(theta), c='k', s=3)
ax[2].scatter(1000*X_[:,0]*np.cos(theta), 1000*X_[:,0]*np.sin(theta), c='k', s=3)
#ax1.scatter(1000*X[:,0]*np.cos(theta), 1000*X[:,2], c='k')
feats_o = X_[np.where(Ri.T)[1]]
feats_i = X_[np.where(Ro.T)[1]]
x_o = 1000*feats_o[:,0]*np.cos(feats_o[:,1])
x_i = 1000*feats_i[:,0]*np.cos(feats_i[:,1])
y_o = 1000*feats_o[:,0]*np.sin(feats_o[:,1])
y_i = 1000*feats_i[:,0]*np.sin(feats_i[:,1])

for j in range(y.shape[0]):
	hid_bo =find_hit_id(bo[j])
	track_id = labels[hid_bo]
	track_id_tr = labels_tr[hid_bo]
	
	if (track_id > -1):
		seg_args = dict(c='C'+str(track_id), alpha=preds[j,0])
		ax[0].plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)

	if (track_id_tr > -1) and y[j]:
		seg_args = dict(c='C'+str(track_id_tr), alpha=preds[j,1])
		ax[1].plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)
	seg_args = dict(c='darkblue', alpha=y[j])
	ax[2].plot([x_o[j],x_i[j]],[y_o[j],y_i[j]], '-', **seg_args)


ax[0].set_xlabel('x [mm]')
ax[0].set_ylabel('y [mm]')
ax[1].set_xlabel('x [mm]')
ax[1].set_ylabel('y [mm]')
ax[2].set_xlabel('x [mm]')
ax[2].set_ylabel('y [mm]')
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[2].set_aspect('equal')
ax[0].set_title('Reconstructed tracks with DBSCAN (predictions)')
ax[1].set_title('Reconstructed tracks with DBSCAN (truth)')
ax[1].set_title('All true edges without reconstruction')

plt.savefig('png/test.png')
