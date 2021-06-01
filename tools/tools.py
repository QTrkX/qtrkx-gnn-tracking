import tools
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
#internal
import os, sys, glob, yaml, datetime, argparse
import csv
import tensorflow as tf

Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])

class GraphDataset():
    def __init__(self, input_dir, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('event') and f.endswith('.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)

def get_dataset(input_dir,n_files):
    return GraphDataset(input_dir, n_files)
def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))
def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y)

def parse_args():
    # generic parser, nothing fancy here
    parser = argparse.ArgumentParser(description='Load config file!')
    add_arg = parser.add_argument
    add_arg('config')
    add_arg('RID')
    return parser.parse_args()
def load_config(args):
    # read the config file 
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if len(glob.glob(config['log_dir']))==0:
            os.mkdir(config['log_dir'])
        # append RID to log dir
        config['log_dir'] = config['log_dir']+'run{}/'.format(args.RID)
        if len(glob.glob(config['log_dir']))==0:
            os.mkdir(config['log_dir'])
        # print all configs
        print('Printing configs: ')
        for key in config:
            print(key + ': ' + str(config[key]))
        print('Log dir: ' + config['log_dir'])
        print('Training data input dir: ' + config['train_dir'])
        print('Validation data input dir: ' + config['train_dir'])
        if config['run_type'] == 'new_run':
            delete_all_logs(config['log_dir'])
    # LOG the config every time
    with open(config['log_dir'] + 'config.yaml', 'w') as f:
        for key in config:
            f.write('%s : %s \n' %(key,str(config[key])))
    # return the config dictionary
    return config

def delete_all_logs(log_dir):
# Delete all .csv files in directory
    log_list = os.listdir(log_dir)
    for item in log_list:
        if item.endswith('.csv'):
            os.remove(log_dir+item)
            print(str(datetime.datetime.now()) + ' Deleted old log: ' + log_dir+item)
    init_all_logs(log_dir)

def init_all_logs(log_dir):
    with open(log_dir+'log_validation.csv', 'a') as f: 
        f.write('accuracy,auc,loss,precision,accuracy_3,precision_3,recall_3,f1_3,accuracy_5,precision_5,recall_5,f1_5,accuracy_7,precision_7,recall_7,f1_7,duration\n')
    with open(log_dir+'log_training.csv', 'a') as f: 
        f.write('accuracy,auc,loss,precision,accuracy_3,precision_3,recall_3,f1_3,accuracy_5,precision_5,recall_5,f1_5,accuracy_7,precision_7,recall_7,f1_7,duration\n')
    with open(log_dir+'summary.csv', 'a') as f:
        f.write('epoch,batch,loss,duration\n')


def log_parameters(log_dir, parameters):
    for idx, params in enumerate(parameters):
        params = params.numpy().flatten()
        with open(log_dir+'log_parameters_%d.csv'%idx, 'a') as f:
            for idy, item in enumerate(params):
                f.write('%f'%item  )
                if (idy+1)!=len(params):
                    f.write(', ')
            f.write('\n')

def log_gradients(log_dir, gradients):
    for idx, grads in enumerate(gradients):
        grads = grads.numpy().flatten()
        with open(log_dir+'log_gradients_%d.csv'%idx, 'a') as f:
            for idy, item in enumerate(grads):
                f.write('%f'%item  )
                if (idy+1)!=len(grads):
                    f.write(', ')
            f.write('\n')

def map2angle(arr0):
    # Mapping the cylindrical coordinates to [0,1]
    arr = np.zeros(arr0.shape, dtype=np.float32)
    r_min     = 0.
    r_max     = 1.1
    arr[:,0] = (arr0[:,0]-r_min)/(r_max-r_min)    



    if (tools.config['dataset'] == 'mu200') or (tools.config['dataset'] == 'mu200_full'):
        phi_min   = -1.0
        phi_max   = 1.0
        arr[:,1]  = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
        z_min     = 0
        z_max     = 1.1
        arr[:,2]  = (np.abs(arr0[:,2])-z_min)/(z_max-z_min)  # take abs of z due to symmetry of z

    elif tools.config['dataset'] == 'mu200_1pT':
        phi_max  = 1.
        phi_min  = -phi_max
        z_max    = 1.1
        z_min    = -z_max
        arr[:,1] = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
        arr[:,2] = (arr0[:,2]-z_min)/(z_max-z_min) 

    elif (tools.config['dataset'] == 'mu10') or (tools.config['dataset'] == 'mu10_big'):
        phi_min   = -1.0
        phi_max   = 1.0
        arr[:,1] = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
        z_min     = -1.1
        z_max     = 1.1
        arr[:,2] = (arr0[:,2]-z_min)/(z_max-z_min) 

    mapping_check(arr)
    return arr
############################################################################################
def mapping_check(arr):
# check if every element of the input array is within limits [0,2*pi]
    for row in arr:
        for item in row:
            if (item > 1) or (item < 0):
                raise ValueError('WARNING!: WRONG MAPPING!!!!!!')

def find_layer(arr):
    layers = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        if arr[i,0] < 5e-2: 
            layers[i] = 0
        elif (arr[i,0] >= 5e-2) and (arr[i,0] < 9e-2):
            layers[i] = 1
        elif (arr[i,0] >= 9e-2) and (arr[i,0] < 15e-2):
            layers[i] = 2
        elif (arr[i,0] >= 15e-2) and (arr[i,0] < 2e-1):
            layers[i] = 3 
        elif (arr[i,0] >= 2e-1) and (arr[i,0] < 3e-1):
            layers[i] = 4
        elif (arr[i,0] >= 3e-1) and (arr[i,0] < 4e-1):
            layers[i] = 5
        elif (arr[i,0] >= 4e-1) and (arr[i,0] < 6e-1):
            layers[i] = 6
        elif (arr[i,0] >= 6e-1) and (arr[i,0] < 7.5e-1):
            layers[i] = 7
        elif (arr[i,0] >= 7.5e-1) and (arr[i,0] < 9e-1):
            layers[i] = 8
        elif (arr[i,0] >= 9e-1) and (arr[i,0] < 10.5e-1):
            layers[i] = 9
        else:
            raise ValueError()
    return layers

def true_fake_weights(labels):
    ''' 
    [weight of fake edges, weight of true edges]
    
    weights are calculated using scripts/print_dataset_specs.py

    '''
    if tools.config['dataset'] == 'mu200':
        weight_list = [1.102973565242351, 0.9146118742361756]
    elif tools.config['dataset'] == 'mu200_1pT':
        weight_list = [1.024985997012696, 0.9762031776515252]
    elif tools.config['dataset'] == 'mu200_full':
        weight_list = [0.5424779619482216, 6.385404773061769]
    elif tools.config['dataset'] == 'mu10':
        weight_list = [3.030203859885135, 0.5988062677334424]
    elif tools.config['dataset'] == 'mu10_big':
        weight_list = [0.9369978711656622, 1.0720851667609774]
    else:
        raise ValueError('dataset not defined')

    return [weight_list[int(labels[i])] for i in range(labels.shape[0])]

def load_params(model, log_path):
    n_layers = len(glob.glob('{}*{}*'.format(log_path,'parameters')))
    if n_layers > 0:
        for idx in range(n_layers):
            param_file = log_path + 'log_parameters_%d'%(idx) + '.csv'
            val_file   = log_path + 'log_validation.csv'
            # read the last line of the parameter file
            with open(param_file, 'r') as f:
                reader = csv.reader(f, delimiter=',')  
                params = np.array(list(reader)).astype(float)[-1]
            with open(val_file, 'r') as f:
                reader = csv.reader(f, delimiter=',')  
                val = np.array(list(reader))
            last_epoch = val.shape[0]-2
            # make sure they have the same shape
            params = np.resize(params, model.trainable_variables[idx].shape)
            # Load the parameter to corresponding layer
            model.trainable_variables[idx].assign(params)
        return model, last_epoch
    else:
        raise ValueError('No parameter log found!')

def get_value_from_dicts(dicts, label, network_label=None):
    ''' Returns the values as an array from a dictionary of dictionaries
    Args:
        dicts (dict): dict of dicts

        label (string): target label

        network_label (string): target network label
    
    Retruns:
        arr (list): list of the values of the corresponding label 
    '''
    arr = []
    if network_label!=None:
        for v in dicts.values():
            arr.append(v[network_label][label])       
    else:
        for v in dicts.values():
            arr.append(v[label])

    return arr

def get_configs(log_path_list):
    configs_dict = {}
    for path in log_path_list:
        # read the config file 
        n_runs = get_n_runs(path)
        path_ = path + 'run1/config.yaml'
        with open(path_, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        # append n_runs to config
        config['n_runs'] = n_runs
        # save config to configs_dict
        configs_dict[path] = config

    # return the configs dictionary
    return configs_dict

def get_n_runs(path):
    n_folders = 0
    for _, dirnames, _ in os.walk(path):
        n_folders += len(dirnames)
    return n_folders