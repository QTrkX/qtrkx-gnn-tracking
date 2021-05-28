import sys, os, time, datetime, csv
sys.path.append(os.path.abspath(os.path.join('.')))
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
'''
n_q = np.array([4,8,16,32])
n_l = np.arange(1,10)
n_h = np.arange(1,100)

# n_hid = 1 plot 
mps     = 26*n_q + 6
circ_10 = 28*n_q + 6
hybrid  = 27*n_q + 6

fig, ax = plt.subplots(1,3,figsize = (16,5),sharey=True, tight_layout=True)

ax[0].plot(n_q, mps, label='mps, ttn')
ax[0].plot(n_q, circ_10, label='qc10')
ax[0].plot(n_q, hybrid, label='hybrid')

ax[0].set_xlabel('number of qubits')
ax[0].set_ylabel('number of parameters')
#ax[0].set_title('Reconstructed tracks with DBSCAN (predictions)')

# constant Nqubits = 4
# constant Nhid = 1 
for q in n_q:
    circ_10 = 5*1 + q*(20+2*n_l+6*1)+1 
    hybrid  = 5*1 + q*(20+1*n_l+6*1)+1 

    ax[1].plot(n_l, circ_10, label='qc10', color='darkred')
    ax[1].plot(n_l, hybrid, label='hybrid', color='darkblue')

ax[1].set_xlabel('number of layers')
ax[1].set_ylabel('number of parameters')

'''
fig, ax = plt.subplots(1,1,figsize = (6,5),tight_layout=True)

max_hid = 100
min_hid = 1
max_qubit  = 16
min_qubit  = 4
max_layers = 5

n_h = (np.arange(max_hid-min_hid+1)+min_hid)

classical = 6*(n_h**2) + 23*n_h + 1  
ax.plot(n_h, classical, label='classical', color='darkorange')

range_ = (np.arange(100)+1)/50 # 100 steps between [0,2]

n_q = np.rint(min_qubit+(n_h-1)*(max_qubit-min_qubit)/(max_hid-min_hid))
n_l = np.rint(1+(n_h-1)*(max_layers-1)/(max_hid-min_hid))
for i in range(max_hid-min_hid+1):
    print('hid. dim: {}, n_qubits: {}, n_layers: {}'.format(n_h[i],n_q[i],n_l[i]))
hybrid = 5*n_h + n_q*(20+2*n_l+6*n_h)+1 
ax.plot(n_h, hybrid, label='hybrid_linear', color='darkblue')


n_q = np.rint(min_qubit+((max_qubit-min_qubit)/(np.log10(max_hid/min_hid)))*np.log10(n_h))
n_l = np.rint(1+((max_layers-1)/(np.log10(max_hid/min_hid)))*np.log10(n_h))
hybrid_log = 5*n_h + n_q*(20+2*n_l+6*n_h)+1

for i in range(max_hid-min_hid+1):
    print('hid. dim: {}, n_qubits: {}, n_layers: {}'.format(n_h[i],n_q[i],n_l[i]))

ax.plot(n_h, hybrid_log, label='hybrid_log', color='darkred')


ax.set_xlabel('hidden dimension size')
ax.set_ylabel('number of parameters')

ax.set_yscale('log')

ax.legend()
ax.grid()

plt.savefig('png/param_scaling/scaling.png')