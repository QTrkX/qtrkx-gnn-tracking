import matplotlib.pyplot as plt 
import matplotlib
import numpy as np
import csv
from sklearn import metrics
import sys, os, glob
import pandas as pd


def label_grabber(comparison_type, log_list):
	path_list = np.char.split(log_path_list,sep='/')
	label_list = [item[-2] for item in path_list]
	labels = np.char.split(label_list, sep='_')
	circs = [r'circuit = '+item[0] for item in labels]
	dims = [r'$N_{hid} = $' +item[1][3:] for item in labels]
	its = [r'$N_{it} = $' +item[2][2:] for item in labels]
	layers = [r'$N_{layer} = $' +item[3][5:] for item in labels]
	
	if comparison_type == 'circuit_comparison':
		label_list = circs 
		title_extension = '(' +dims[0]+', '+its[0]+', '+layers[0]+')'
	elif comparison_type == 'dimension_comparison':
		label_list =  dims 
		title_extension = '(' +circs[0]+', '+its[0]+', '+layers[0]+')'
	elif comparison_type == 'iteration_comparison':
		label_list = its 
		title_extension = '(' +circs[0]+', '+dims[0]+', '+layers[0]+')'
	elif comparison_type == 'layer_comparison':
		label_list = layers
		title_extension = '(' +circs[0]+', '+dims[0]+', '+its[0]+')'
	else:
		raise ValueError('comparison_type not defined!')
	
	return label_list, title_extension


def n_items_min(log_list):
	n_list = []
	for fname in log_list:
		with open(fname+'log_validation.csv') as f:
			for i, l in enumerate(f):
				pass
		n_list.append(i+1)
	return min(n_list)

def get_n_runs(path):
	n_folders = 0
	for _, dirnames, _ in os.walk(path):
		n_folders += len(dirnames)
	return n_folders

def create_folder(folderpath):
	if len(glob.glob(folderpath))==0:
		# create if it doesn't exist
		os.mkdir(folderpath)
		print('New folder created at %s'%folderpath)

def get_multiple_entries(path, attribute_id):
	'''returns the same column from multiple files
	0: accuracy
	1: auc
	2: loss
	3: precision
	'''
	n_runs = get_n_runs(path)
	log_list = [path + 'run' + str(i+1) + '/' for i in range(n_runs)]
	n_items = n_items_min(log_list)
	arr  = np.empty(shape=(n_runs,n_items))
	for i in range(n_runs):
		with open(log_list[i]+'log_validation.csv', 'r') as f:
			reader = csv.reader(f, delimiter=',')  
			validation = np.array(list(reader)).astype(float)
			arr[i,:] = validation[0:n_items,attribute_id]	

	x = [i*interval for i  in range(n_items)]
	return arr, x


log_path_list = [
	'logs/15Z_embedding/15_hid1_it1_layer1/',
	'logs/15Z_embedding/14_hid1_it1_layer1/',
	'logs/15Z_embedding/11_hid1_it1_layer1/',
	'logs/15Z_embedding/10_hid1_it1_layer1/',
	'logs/15Z_embedding/7_hid1_it1_layer1/',
	'logs/15Z_embedding/3_hid1_it1_layer1/',
	]

pdf_location = 'pdf/15Z_embedding/circuit_comparison/'
png_location = 'png/15Z_embedding/circuit_comparison/'

interval = 50

from mpl_toolkits.mplot3d import Axes3D


metrics = pd.read_csv('logs/15Z_embedding/circuit_metrics_zero_inp.csv')

auc_list = np.empty(shape=(len(log_path_list),3))   
for i, path in enumerate(log_path_list):
	arr, x = get_multiple_entries(path, 2)#loss
	auc_list[i] = arr.min(axis=1)

auc_mean = np.mean(auc_list,axis=1)
auc_err  = np.std(auc_list,axis=1)

circ_list = ['15','14','11','10','7','3']

fig, ax = plt.subplots(1, figsize = (6,5), tight_layout=True)

losses = [0.2315,0.1326,0.1362,0.0980,0.1416,0.1930]
losses_err = [0.1013,0.0412,0.0438,0.0463,0.0385,0.0474]
fr = [11.8672,17.1091,17.5197,14.3970,21.5534,15.6554]
fr_err = [4.0603, 6.5251, 5.7902, 5.3096, 7.0815, 6.4685] 
ax.errorbar(fr,losses,losses_err,xerr=fr_err,linestyle="None",marker="o",color='darkred')
for idx,circ in enumerate(circ_list):
	ax.text(fr[idx], losses[idx],"{}".format(circ))
#ax.set_xlim(np.min(metrics['exp'])*0.9,np.max(metrics['exp'])*1.1)
#ax.set_ylim(np.min(auc_mean)*0.9,np.max(auc_mean)*1.1)
#ax[0,0].set_xscale('log')
#ax.ticklabel_format(axis="x", style="sci",scilimits=(0,0))
ax.set_xlabel('Fisher-Rao Norm')
ax.set_ylabel('Min. Loss')
ax.grid()

plt.show()

fig, ax = plt.subplots(2, 2, figsize = (11,7), sharey=True, tight_layout=True)

ax[0,0].errorbar(metrics['exp'],auc_mean,auc_err,xerr=metrics['exp_err'],linestyle="None",marker="o",color='darkred')
for idx,circ in enumerate(circ_list):
	ax[0,0].text(metrics['exp'][idx], auc_mean[idx],"{}".format(circ))
ax[0,0].set_xlim(np.min(metrics['exp'])*0.9,np.max(metrics['exp'])*1.1)
ax[0,0].set_ylim(np.min(auc_mean)*0.9,np.max(auc_mean)*1.1)
#ax[0,0].set_xscale('log')
ax[0,0].ticklabel_format(axis="x", style="sci",scilimits=(0,0))
ax[0,0].set_xlabel('Expressibility (KL Divergences)')
ax[0,0].set_ylabel('Min. Loss')
ax[0,0].grid()


ax[0,1].errorbar(metrics['ent'],auc_mean,auc_err,xerr=metrics['ent_err'],linestyle="None",marker="o",color='darkred')
for idx,circ in enumerate(circ_list):
	ax[0,1].text(metrics['ent'][idx], auc_mean[idx],"{}".format(circ))
ax[0,1].set_xlim(np.min(metrics['ent'])*0.9,np.max(metrics['ent'])*1.1)
ax[0,1].set_ylim(np.min(auc_mean)*0.9,np.max(auc_mean)*1.1)
ax[0,1].set_xlabel('Entangling capability')
#ax[0,1].set_xscale('log')
ax[0,1].ticklabel_format(axis="x", style="sci",scilimits=(0,0))
ax[0,1].grid()

ax[1,0].errorbar(metrics['n_params'],auc_mean,auc_err,linestyle="None",marker="o",color='darkred')
for idx,circ in enumerate(circ_list):
	ax[1,0].text(metrics['n_params'][idx], auc_mean[idx],"{}".format(circ))
ax[1,0].set_xlim(np.min(metrics['n_params'])*0.9,np.max(metrics['n_params'])*1.1)
ax[1,0].set_ylim(np.min(auc_mean)*0.9,np.max(auc_mean)*1.1)
ax[1,0].set_xlabel('Number of PQC Parameters')
ax[1,0].set_ylabel('Min. Loss')
ax[1,0].grid()

ax[1,1].errorbar(metrics['rel_ent'],auc_mean,auc_err,xerr=metrics['rel_ent_err'],linestyle="None",marker="o",color='darkred')
for idx,circ in enumerate(circ_list):
	ax[1,1].text(metrics['rel_ent'][idx], auc_mean[idx],"{}".format(circ))
ax[1,1].set_xlim(np.min(metrics['rel_ent'])*0.9,np.max(metrics['rel_ent'])*1.1)
ax[1,1].set_ylim(np.min(auc_mean)*0.9,np.max(auc_mean)*1.1)
#ax[3].set_xscale('log')
#ax[0].ticklabel_format(axis="x", style="sci",scilimits=(0,0))
ax[1,1].set_xlabel('Relative Entropy')
ax[1,1].grid()


#plt.show()
plt.savefig(pdf_location+'metric_comparison_all_zero_inp.pdf')
plt.savefig(png_location+'metric_comparison_all_zero_inp.png')

fig, ax = plt.subplots(1,2, figsize = (11,5),tight_layout=True)
ax[0].errorbar(metrics['exp']/18.347, metrics['rel_ent'], xerr=metrics['exp_err']/18.347, yerr=metrics['rel_ent_err'] , linestyle="None",marker=None,zorder=0)
a = ax[0].scatter(metrics['exp']/18.347, metrics['rel_ent'], c=auc_mean, marker="o",cmap='magma',zorder=100)
for idx,circ in enumerate(circ_list):
	ax[0].text(metrics['exp'][idx]/18.347, metrics['rel_ent'][idx],"{}".format(circ))
ax[0].set_xlabel('Expressibility (KL Divergences)')
ax[0].set_ylabel('Relative Entropy')
ax[0].grid()
ax[1].errorbar(metrics['exp']/18.347, metrics['ent'], xerr=metrics['exp_err']/18.347, yerr=metrics['ent_err'] , linestyle="None",marker=None,zorder=0)
a = ax[1].scatter(metrics['exp']/18.347, metrics['ent'], c=auc_mean, marker="o",cmap='magma',zorder=100 )
for idx,circ in enumerate(circ_list):
	ax[1].text(metrics['exp'][idx]/18.347, metrics['ent'][idx],"{}".format(circ))
ax[1].set_xlabel('Expressibility (KL Divergences)')
ax[1].set_ylabel('Entangling capability')
ax[1].grid()
fig.colorbar(a)
#plt.show()


plt.savefig(pdf_location+'metric_comparison_2d_zero_inp.pdf')
plt.savefig(png_location+'metric_comparison_2d_zero_inp.png')

	