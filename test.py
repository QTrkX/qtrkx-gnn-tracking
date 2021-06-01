import sys
import os
import time
import datetime
import numpy as np
from sklearn import metrics
from tools.tools import *
import tensorflow as tf

def test(config, model, test_type):
    print(
        str(datetime.datetime.now()) 
        + ' Starting testing the %s set with '%(test_type)
        + str(config['n_valid']) + ' subgraphs!'
        )

    # Start timer
    t_start = time.time()
    
    # load data
    if test_type == 'valid':
        valid_data = get_dataset(config['valid_dir'], config['n_valid'])
        n_test = config['n_valid']
        log_extension = 'validation'
    elif test_type == 'train':
        valid_data = get_dataset(config['train_dir'], config['n_train'])
        n_test = config['n_train']
        log_extension = 'training'

    # Load loss function
    loss_fn = getattr(tf.keras.losses, config['loss_func'])()

    # Obtain predictions and labels
    for n in range(n_test):

        X, Ri, Ro, y = valid_data[n]

        if n == 0:
            preds = model([map2angle(X), Ri, Ro])
            labels = y
        else:	
            out = model([map2angle(X), Ri, Ro])
            preds  = tf.concat([preds, out], axis=0)
            labels = tf.concat([labels, y], axis=0)

    labels = tf.reshape(labels, shape=(labels.shape[0],1))

    # calculate weight for each edge to avoid class imbalance
    weights = tf.convert_to_tensor(true_fake_weights(labels))

    loss = loss_fn(labels, preds, sample_weight=weights).numpy()

    # Log all predictons (takes some considerable time - use only for debugging)
    if config['log_verbosity']>=3 and test_type=='valid':	
        with open(config['log_dir']+'log_validation_preds.csv', 'a') as f:
            for i in range(len(preds)):
                f.write('%.4f, %.4f\n' %(preds[i],labels[i]))

    # Calculate Metrics
    # To Do: add 0.8 threshold and other possible metrics
    # efficency, purity etc.
    labels = labels.numpy()
    preds  = preds.numpy()

    #n_edges = labels.shape[0]
    #n_class = [n_edges - sum(labels), sum(labels)]

    fpr, tpr, _ = metrics.roc_curve(labels.astype(int),preds,pos_label=1 )
    auc                = metrics.auc(fpr,tpr)

    tn, fp, fn, tp = metrics.confusion_matrix(
        labels.astype(int),(preds > 0.3)*1
        ).ravel() # get the confusion matrix for 0.3 threshold
    accuracy_3  = (tp+tn)/(tn+fp+fn+tp)
    precision_3 = tp/(tp+fp) # also named purity
    recall_3    = tp/(tp+fn) # also named efficiency
    f1_3        = (2*precision_3*recall_3)/(precision_3+recall_3) 

    tn, fp, fn, tp = metrics.confusion_matrix(
        labels.astype(int),(preds > 0.5)*1
        ).ravel() # get the confusion matrix for 0.5 threshold
    accuracy_5  = (tp+tn)/(tn+fp+fn+tp)
    precision_5 = tp/(tp+fp) # also named purity
    recall_5    = tp/(tp+fn) # also named efficiency
    f1_5        = (2*precision_5*recall_5)/(precision_5+recall_5) 

    tn, fp, fn, tp = metrics.confusion_matrix(
        labels.astype(int),(preds > 0.7)*1
        ).ravel() # get the confusion matrix for 0.7 threshold
    accuracy_7  = (tp+tn)/(tn+fp+fn+tp)
    precision_7 = tp/(tp+fp) # also named purity
    recall_7    = tp/(tp+fn) # also named efficiency
    f1_7        = (2*precision_7*recall_7)/(precision_7+recall_7) 

    # End timer
    duration = time.time() - t_start

    # Log Metrics
    with open(config['log_dir']+'log_'+log_extension+'.csv', 'a') as f:
        f.write('%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %d\n' %(accuracy_5, auc, loss, precision_5, accuracy_3, precision_3, recall_3, f1_3, accuracy_5, precision_5, recall_5, f1_5, accuracy_7, precision_7, recall_7, f1_7, duration))

    # Print summary
    print(str(datetime.datetime.now()) + ': ' + log_extension+' Test:  Loss: %.4f,  AUC: %.4f, Acc: %.4f,  Precision: %.4f -- Elapsed: %dm%ds' %(loss, auc, accuracy_5*100, precision_5, duration/60, duration%60))

    del labels
    del preds
